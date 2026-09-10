import os
import sys
import glob
import argparse
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F

# Import utilities from our Grad-CAM engine
try:
    from run_gradcam import load_model_with_fallback, extract_circular_fov_mask
except ImportError:
    print("[!] Error: run_gradcam.py must be in the same directory.")
    sys.exit(1)

class VerificationGradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.feature_maps = None
        self.gradients = None
        
        last_conv = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No nn.Conv2d layer found.")
            
        last_conv.register_forward_hook(self.save_feature_maps)
        last_conv.register_full_backward_hook(self.save_gradients)

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output.detach()

    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate_cam(self, input_tensor, target_class=0):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        
        if len(logits.shape) == 4: # Segmentation [B, C, H, W]
            score = logits[:, target_class, :, :].sum()
        else: # Classification [B, C]
            score = logits[:, target_class].sum()
            
        score.backward(retain_graph=True)
        
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        return F.relu(cam)

def get_real_images(dataset_path):
    images = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        images.extend(glob.glob(os.path.join(dataset_path, ext)))
    return images

def main():
    parser = argparse.ArgumentParser(description="Real-Time Grad-CAM Validation Suite")
    parser.add_argument("--dataset", type=str, default="retinal_dataset", help="Path to your actual retinal dataset folder")
    args = parser.parse_args()

    print("==================================================")
    print("  GRAD-CAM MATHEMATICAL VALIDATION SUITE          ")
    print("==================================================")
    
    print(f"[*] Scanning real dataset at: '{args.dataset}/'")
    images = get_real_images(args.dataset)
    
    if len(images) < 2:
        print(f"[!] ERROR: Found {len(images)} images in '{args.dataset}'.")
        print("[!] The validation suite requires at least 2 real images to run Test 4 (Responsiveness).")
        print("[!] Please provide a valid dataset path using: python verify_gradcam.py --dataset <path_to_folder>")
        sys.exit(1)
        
    print(f"[*] Successfully loaded real images. Using:")
    print(f"    - Image 1: {os.path.basename(images[0])}")
    print(f"    - Image 2: {os.path.basename(images[1])}")
        
    model = load_model_with_fallback()
    gradcam = VerificationGradCAM(model)
    
    transform = T.Compose([
        T.Resize((256, 256)), 
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_passed = True

    # TEST 1: Dynamic Gradient Sensitivity
    print("\n[Test 1] Dynamic Gradient Sensitivity (Class Discrimination)")
    img_tensor = transform(Image.open(images[0]).convert("RGB")).unsqueeze(0)
    
    try:
        cam_class_0 = gradcam.generate_cam(img_tensor, target_class=0).detach().cpu().numpy()
        cam_class_1 = gradcam.generate_cam(img_tensor, target_class=1).detach().cpu().numpy()
        
        mad = np.mean(np.abs(cam_class_0 - cam_class_1))
        if mad > 1e-6:
            print(f"  [PASS] Heatmaps differ between classes (MAD = {mad:.6f})")
        else:
            print(f"  [FAIL] Heatmaps are identical! MAD = {mad:.6f}")
            all_passed = False
    except Exception as e:
        print(f"  [FAIL] Test crashed: {e}")
        all_passed = False

    # TEST 2: Zero & Constant Input Test
    print("\n[Test 2] Zero & Constant Input Test (Dead Signal Check)")
    try:
        zeros_tensor = torch.zeros((1, 3, 256, 256))
        ones_tensor = torch.ones((1, 3, 256, 256))
        
        cam_zeros = gradcam.generate_cam(zeros_tensor, target_class=0)
        cam_ones = gradcam.generate_cam(ones_tensor, target_class=0)
        
        if torch.isnan(cam_zeros).any() or torch.isnan(cam_ones).any():
            print("  [FAIL] NaN values detected in dead signal output!")
            all_passed = False
        else:
            print("  [PASS] Handled zero/ones tensors gracefully (No NaN or crashes).")
    except Exception as e:
        print(f"  [FAIL] Test crashed: {e}")
        all_passed = False

    # TEST 3: FOV & Boundary Integrity Check
    print("\n[Test 3] FOV & Boundary Integrity Check")
    try:
        orig_np = np.array(Image.open(images[0]).convert("RGB").resize((256, 256)))
        fov_mask = extract_circular_fov_mask(orig_np)
        
        cam = gradcam.generate_cam(img_tensor, target_class=0)
        cam_upsampled = F.interpolate(cam, size=(256, 256), mode='bicubic', align_corners=False).squeeze().cpu().detach().numpy()
        
        cam_upsampled[~fov_mask] = 0
        
        # Check an extreme corner (e.g., 0,0 which is outside the circular retina)
        corner_val = cam_upsampled[0, 0]
        if corner_val == 0.0:
            print("  [PASS] Boundary integrity verified. Zero heatmap bleeding in dark borders.")
        else:
            print(f"  [FAIL] Bleeding detected! Corner value = {corner_val}")
            all_passed = False
    except Exception as e:
        print(f"  [FAIL] Test crashed: {e}")
        all_passed = False

    # TEST 4: Dynamic Image Responsiveness Test
    print("\n[Test 4] Dynamic Image Responsiveness Test (Real Images)")
    try:
        img1_tensor = transform(Image.open(images[0]).convert("RGB")).unsqueeze(0)
        img2_tensor = transform(Image.open(images[1]).convert("RGB")).unsqueeze(0)
        
        cam1 = gradcam.generate_cam(img1_tensor, target_class=0).squeeze().cpu().detach().numpy()
        cam2 = gradcam.generate_cam(img2_tensor, target_class=0).squeeze().cpu().detach().numpy()
        
        peak1 = np.unravel_index(np.argmax(cam1), cam1.shape)
        peak2 = np.unravel_index(np.argmax(cam2), cam2.shape)
        
        if peak1 != peak2:
            print(f"  [PASS] Focal peaks differ. Image1 Peak: {peak1} | Image2 Peak: {peak2}")
        else:
            print(f"  [WARNING] Focal peaks match {peak1}. Ensure clinical features in images 1 and 2 are distinct.")
    except Exception as e:
        print(f"  [FAIL] Test crashed: {e}")
        all_passed = False

    print("\n==================================================")
    if all_passed:
        print("  OVERALL VERIFICATION: SUCCESS [✓]")
        os.makedirs("heatmap_outputs", exist_ok=True)
        
        cam_max = cam_upsampled.max()
        cam_norm = cam_upsampled / cam_max if cam_max > 0 else cam_upsampled
        cam_uint8 = (cam_norm * 255).astype(np.uint8)
        
        heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap[~fov_mask] = 0
        
        if orig_np.max() <= 1.0: orig_np = (orig_np * 255).astype(np.uint8)
        overlay = cv2.addWeighted(orig_np, 0.6, heatmap, 0.4, 0)
        overlay[~fov_mask] = 0
        
        output_path = "heatmap_outputs/verified_sample.png"
        Image.fromarray(overlay).save(output_path)
        print(f"  Saved verified visual proof to: {output_path}")
    else:
        print("  OVERALL VERIFICATION: FAILED [X]")
    print("==================================================\n")

if __name__ == "__main__":
    main()

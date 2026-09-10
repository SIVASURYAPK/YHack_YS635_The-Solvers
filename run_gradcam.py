import os
import glob
import argparse
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T

# Fallback/Model imports
try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None
from torchvision.models import resnet50, ResNet50_Weights

class ClinicalGradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.feature_maps = None
        self.gradients = None
        
        self.target_layer, self.target_name = self._find_last_conv_layer(self.model)
        if self.target_layer is None:
            raise ValueError("[!] Error: No nn.Conv2d layer found.")
            
        self.target_layer.register_forward_hook(self.save_feature_maps)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def _find_last_conv_layer(self, model):
        last_conv = None
        target_name = None
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer4'):
            return model.encoder.layer4[-1], "encoder.layer4[-1]"
            
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
                target_name = name
        return last_conv, target_name

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output.detach()

    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate_cam(self, input_tensor, target_class_idx):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        
        if len(logits.shape) == 4: # Segmentation [B, C, H, W]
            score = logits[:, target_class_idx, :, :].sum()
        else: # Classification [B, C]
            score = logits[:, target_class_idx].sum()
            
        score.backward(retain_graph=True)
        
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        return F.relu(cam)

def extract_circular_fov_mask(image_np, threshold=15):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        return clean_mask > 0
    return mask > 0

# [P2 FIX] Targeted Checkpoint Loading & [P1 FIX] Device Mapping
def load_model_with_fallback(preferred_weights="idrid_finetuned.pth", device="cpu"):
    target_path = None
    
    # Strictly check for the preferred weights first
    if os.path.exists(preferred_weights):
        target_path = preferred_weights
    else:
        # Fallback to any .pth file if the preferred one is missing
        pth_files = glob.glob("*.pth")
        if pth_files:
            target_path = pth_files[0]

    if target_path:
        print(f"[*] Found teammate checkpoint: {target_path}")
        
        if smp is None:
            print("="*60)
            print("[!] CRITICAL WARNING: 'segmentation_models_pytorch' is NOT installed.")
            print("[!] The script CANNOT load your trained U-Net weights without it.")
            print("[!] Please run: pip install segmentation-models-pytorch")
            print("="*60)
            print("[*] Falling back to untrained ResNet-50 for demonstration only...")
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
            return model.to(device)
            
        print("[*] Rebuilding SMP U-Net (SE-ResNet50) architecture...")
        try:
            model = smp.Unet(encoder_name='se_resnet50', classes=5, in_channels=3)
            # Load weights directly to the designated device
            data = torch.load(target_path, map_location=device)
            sd = data.get('state_dict', data.get('model_state_dict', data)) if isinstance(data, dict) else data
            
            if all(k.startswith('module.') for k in sd.keys()):
                sd = {k.replace('module.', ''): v for k, v in sd.items()}
                
            model.load_state_dict(sd, strict=False)
            return model.to(device)
        except Exception as e:
            print(f"[!] Failed to load trained U-Net weights: {e}")
            
    print("[*] Loading standard torchvision ResNet-50 fallback...")
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    return model.to(device)

def process_and_save_gradcam(model, image_path, target_class_idx, fov_threshold=15, output_dir="heatmap_outputs", device="cpu"):
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_class{target_class_idx}_heatmap.png")

    print(f"[*] Processing: {image_path} | Class: {target_class_idx} | FOV Threshold: {fov_threshold}")
    img_pil = Image.open(image_path).convert("RGB")
    orig_np = np.array(img_pil)
    
    fov_mask = extract_circular_fov_mask(orig_np, threshold=fov_threshold)
    
    transform = T.Compose([
        T.Resize((256, 256)), 
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # [P1 FIX] Move input tensor to the designated hardware device
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    gradcam = ClinicalGradCAM(model)
    raw_cam = gradcam.generate_cam(input_tensor, target_class_idx)

    img_h, img_w = orig_np.shape[:2]
    cam_upsampled = F.interpolate(raw_cam, size=(img_h, img_w), mode='bicubic', align_corners=False)
    cam_upsampled = cam_upsampled.squeeze().cpu().numpy()
    
    cam_upsampled = np.maximum(cam_upsampled, 0)
    
    cam_max = cam_upsampled.max()
    cam_norm = cam_upsampled / cam_max if cam_max > 0 else cam_upsampled
    cam_uint8 = (cam_norm * 255).astype(np.uint8)
    
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap[~fov_mask] = 0
    
    if orig_np.max() <= 1.0: 
        orig_np = (orig_np * 255).astype(np.uint8)
        
    overlay = cv2.addWeighted(orig_np, 0.5, heatmap, 0.5, 0)
    overlay[~fov_mask] = 0
    
    Image.fromarray(overlay).save(output_path)
    print(f"[*] Success! Saved to: {output_path}")
    return output_path

def main():
    print("--- CLINICAL GRAD-CAM GENERATOR ---")
    parser = argparse.ArgumentParser(description="Clinical Grad-CAM Heatmap Tool")
    parser.add_argument("--image", type=str, help="Path to input retinal image")
    parser.add_argument("--class_idx", type=int, default=1, help="Pathology class index to target (0-4)")
    parser.add_argument("--fov_thresh", type=int, default=15, help="Luminosity threshold for FOV crop (default: 15)")
    parser.add_argument("--weights", type=str, default="idrid_finetuned.pth", help="Specific .pth checkpoint to load")
    args = parser.parse_args()

    # [P1 FIX] Dynamic Hardware Acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Hardware Acceleration Engine: {device}")

    image_path = args.image
    if not image_path:
        image_path = input("Please enter the path to the retinal fundus image: ").strip()

    if not os.path.exists(image_path):
        print(f"[!] Error: Image not found at '{image_path}'.")
        return

    model = load_model_with_fallback(preferred_weights=args.weights, device=device)
    
    process_and_save_gradcam(model, image_path, args.class_idx, fov_threshold=args.fov_thresh, device=device)

if __name__ == "__main__":
    main()

import argparse
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights

class FocusedGradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.feature_maps = None
        self.gradients = None
        
        # Dynamically hook into the final nn.Conv2d layer
        self.target_layer, self.target_name = self._find_last_conv_layer(self.model)
        if self.target_layer is None:
            raise ValueError("[!] Error: No nn.Conv2d layer found.")
            
        print(f"[*] Dynamically hooked into layer: {self.target_name}")
        self.target_layer.register_forward_hook(self.save_feature_maps)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def _find_last_conv_layer(self, model):
        last_conv = None
        target_name = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
                target_name = name
        return last_conv, target_name

    def save_feature_maps(self, module, input, output):
        self.feature_maps = output.detach()

    def save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate_cam(self, input_tensor):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        
        if len(logits.shape) == 4:
            score = logits[:, 0, :, :].sum()
        else:
            score = logits[:, logits.argmax(dim=1)].sum()
            
        score.backward()
        
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        return F.relu(cam)

def extract_fov_mask(image_np, threshold=15):
    """Creates a boolean mask of the actual retina to strictly hide black borders."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    # Morphological closing to fill small internal holes
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask > 0

def main():
    parser = argparse.ArgumentParser(description="Clinically Focused Grad-CAM Generator")
    parser.add_argument("--image", required=True, help="Path to input fundus image")
    parser.add_argument("--output", default="heatmap_result.png", help="Path to save output heatmap")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"[!] Error: Image not found at {args.image}")
        return

    print("[*] Loading Model (Pretrained ResNet-50 Fallback)...")
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    
    print(f"[*] Processing image: {args.image}")
    img_pil = Image.open(args.image).convert("RGB")
    orig_np = np.array(img_pil)
    
    # Extract the Retinal Field of View Mask
    fov_mask = extract_fov_mask(orig_np)
    
    transform = T.Compose([
        T.Resize((256, 256)), 
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img_pil).unsqueeze(0)

    print("[*] Generating Focused Grad-CAM heatmap...")
    gradcam = FocusedGradCAM(model)
    raw_cam = gradcam.generate_cam(input_tensor)

    # Upsample to native resolution
    img_h, img_w = orig_np.shape[:2]
    cam_upsampled = F.interpolate(raw_cam, size=(img_h, img_w), mode='bilinear', align_corners=False)
    cam_upsampled = cam_upsampled.squeeze().cpu().numpy()
    
    # ---------------------------------------------------------
    # FINE-GRAINED FEATURE FOCUS (Percentile Clipping)
    # ---------------------------------------------------------
    # Only keep the top 20% most salient features, suppress the rest to 0
    threshold_val = np.percentile(cam_upsampled, 80)
    cam_upsampled[cam_upsampled < threshold_val] = 0
    
    # Min-Max Normalize the remaining highly salient regions
    cam_max = cam_upsampled.max()
    if cam_max > 0:
        cam_norm = cam_upsampled / cam_max
    else:
        cam_norm = cam_upsampled
        
    cam_uint8 = (cam_norm * 255).astype(np.uint8)
    
    # ---------------------------------------------------------
    # CLEAN VISUALIZATION & FOV MASKING
    # ---------------------------------------------------------
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Suppress heatmap outside the retina so background stays black
    heatmap[~fov_mask] = 0
    
    if orig_np.max() <= 1.0:
        orig_np = (orig_np * 255).astype(np.uint8)
        
    # Alpha Blending (0.6 original + 0.4 heatmap)
    overlay = cv2.addWeighted(orig_np, 0.6, heatmap, 0.4, 0)
    
    # Ensure the final output background remains fully black, not semi-transparent
    overlay[~fov_mask] = 0
    
    Image.fromarray(overlay).save(args.output)
    print(f"[*] Success! Clinically convincing heatmap saved to {args.output}")

if __name__ == "__main__":
    main()

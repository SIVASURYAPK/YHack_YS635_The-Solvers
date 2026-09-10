import cv2
import torch
import numpy as np
import torch.nn.functional as F

def generate_visual_overlay(raw_cam, original_image_np, predicted_mask_np=None, alpha=0.5):
    """
    Upsamples the raw CAM tensor, applies a medical colormap, and blends it with the fundus image.
    
    :param raw_cam: The output CAM tensor from the Grad-CAM engine [1, 1, H', W'].
    :param original_image_np: Original RGB fundus image as numpy array (H, W, 3), range [0, 1] or [0, 255].
    :param predicted_mask_np: (Optional) Binary mask of the prediction to draw contours (H, W).
    :param alpha: Heatmap transparency.
    :return: Overlaid RGB image as numpy array [0, 255] uint8.
    """
    # 1. Standardize original image to [0, 255] uint8
    if original_image_np.max() <= 1.0:
        original_image_np = (original_image_np * 255).astype(np.uint8)
    else:
        original_image_np = original_image_np.astype(np.uint8)
        
    img_h, img_w = original_image_np.shape[:2]

    # 2. Upsample raw CAM to match the original fundus resolution
    cam_upsampled = F.interpolate(raw_cam, size=(img_h, img_w), mode='bilinear', align_corners=False)
    cam_upsampled = cam_upsampled.squeeze().cpu().detach().numpy()
    
    # 3. Min-Max Normalize CAM to [0, 1] then scale to [0, 255] for OpenCV
    cam_min, cam_max = cam_upsampled.min(), cam_upsampled.max()
    if cam_max != cam_min:
        cam_normalized = (cam_upsampled - cam_min) / (cam_max - cam_min)
    else:
        cam_normalized = cam_upsampled
    
    cam_uint8 = (cam_normalized * 255).astype(np.uint8)
    
    # 4. Apply OpenCV Colormap (JET is standard, TURBO is great for medical contrast)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # 5. Blend Heatmap with Original Image
    overlay = cv2.addWeighted(original_image_np, 1 - alpha, heatmap, alpha, 0)
    
    # 6. Add Segmentation Boundary Annotations (For Hackathon Judges)
    if predicted_mask_np is not None:
        # Convert thresholded mask to uint8 {0, 255}
        mask_uint8 = (predicted_mask_np > 0.5).astype(np.uint8) * 255
        # Find spatial contours of the segmented lesions
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Draw bright green outlines around the lesions
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        
    return overlay

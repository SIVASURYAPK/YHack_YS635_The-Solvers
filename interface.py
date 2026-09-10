import io
import base64
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

# Import our custom modules
from unet_gradcam import SegmentationGradCAM
from visualization import generate_visual_overlay

def get_unet_gradcam(model, image, target_class_idx=1, target_size=(512, 512)):
    """
    The single plug-and-play integration hook for the frontend/API teammates.
    
    :param model: The already-loaded PyTorch U-Net model.
    :param image: A PIL.Image or path to the fundus image.
    :param target_class_idx: The lesion channel to explain (e.g., 1=Microaneurysms, 2=Hemorrhages).
    :param target_size: The resolution the model expects.
    :return: A dictionary containing the overlay as Numpy, PIL, and Base64.
    """
    # 1. Load image if path is given
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('RGB')
        
    original_image_np = np.array(image.resize(target_size)) / 255.0

    # 2. Preprocess for SMP ImageNet Encoders
    transform = T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0) # Shape: [1, 3, H, W]
    
    # 3. Generate Raw CAM
    explainer = SegmentationGradCAM(model=model)
    raw_cam, logits = explainer.generate_cam(input_tensor, target_class_idx)
    
    # 4. Extract Predicted Mask (Sigmoid > 0.5 for the specific lesion channel)
    # Using torch.sigmoid because IDRiD models typically use multi-label binary segmentation
    prob_mask = torch.sigmoid(logits[0, target_class_idx]).cpu().detach().numpy()
    
    # 5. Generate Visual Overlay (Heatmap + Contours)
    overlay_np = generate_visual_overlay(raw_cam, original_image_np, prob_mask, alpha=0.5)
    
    # 6. Format Outputs for API / Frontend
    overlay_pil = Image.fromarray(overlay_np)
    
    buffered = io.BytesIO()
    overlay_pil.save(buffered, format="JPEG")
    base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return {
        "numpy": overlay_np,
        "pil": overlay_pil,
        "base64": f"data:image/jpeg;base64,{base64_str}"
    }

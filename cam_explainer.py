import io
import base64
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class CAMExplainer:
    def __init__(self, model, target_layers):
        """
        Initializes the plug-and-play Grad-CAM explainer.
        
        :param model: The PyTorch model (e.g., ResNet, EfficientNet).
        :param target_layers: List of target layers to compute CAM on (e.g., [model.layer4[-1]]).
        """
        self.model = model
        self.target_layers = target_layers
        # Initialize the GradCAM object from the pytorch-grad-cam library
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)

    def generate_cam(self, input_tensor, original_image_np, target_category=None):
        """
        Generates the Grad-CAM heatmap and overlays it on the original image.
        
        :param input_tensor: The preprocessed image tensor (1, C, H, W) ready for the model.
        :param original_image_np: The original image as a NumPy array (H, W, 3) normalized to float32 in [0, 1].
        :param target_category: The target class index. If None, uses the highest scoring category.
        :return: NumPy array of the superimposed image (H, W, 3) in [0, 255] range, uint8 RGB format.
        """
        targets = [ClassifierOutputTarget(target_category)] if target_category is not None else None
        
        # Generate the raw CAM heatmap
        # grayscale_cam shape will be (1, H, W)
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        # Overlay the heatmap on the original image
        # Note: show_cam_on_image expects original image to be float32 in [0, 1]
        visualization = show_cam_on_image(original_image_np, grayscale_cam, use_rgb=True)
        return visualization

    def get_base64_cam(self, input_tensor, original_image_np, target_category=None):
        """
        Generates the CAM overlay and returns it as a base64 encoded string.
        Perfect for returning visual explainability over a FastAPI / Flask endpoint.
        """
        visualization = self.generate_cam(input_tensor, original_image_np, target_category)
        
        # Convert [0, 255] uint8 array to PIL Image
        img_pil = Image.fromarray(visualization)
        
        # Buffer to base64
        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return img_base64

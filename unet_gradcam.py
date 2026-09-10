import torch
import torch.nn.functional as F

class SegmentationGradCAM:
    def __init__(self, model, target_layer=None):
        """
        Zero-dependency PyTorch hook class for U-Net Segmentation Grad-CAM.
        """
        self.model = model
        self.model.eval()
        self.feature_maps = None
        self.gradients = None
        
        # 1. Auto-Layer Fallback (Specifically detects SMP U-Net architecture)
        if target_layer is None:
            if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer4'):
                # Default to the deepest SE-ResNet bottleneck block
                self.target_layer = model.encoder.layer4[-1] 
            elif hasattr(model, 'decoder') and hasattr(model.decoder, 'blocks'):
                # Fallback to the final decoder block
                self.target_layer = model.decoder.blocks[-1]
            else:
                raise ValueError("Could not auto-detect target layer. Please provide it manually.")
        else:
            self.target_layer = target_layer

        # 2. Register Hooks (Hooks are attached without altering original model code)
        self.target_layer.register_forward_hook(self.save_feature_maps)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_feature_maps(self, module, input, output):
        # Capture the forward activations (A)
        self.feature_maps = output.detach()

    def save_gradients(self, module, grad_in, grad_out):
        # Capture the backward gradients (dY/dA)
        self.gradients = grad_out[0].detach()

    def generate_cam(self, input_tensor, target_class_idx):
        """
        Generates the raw CAM tensor for a specific lesion channel.
        """
        self.model.zero_grad()
        
        # Forward pass (Outputs [B, 5, H, W])
        logits = self.model(input_tensor)
        
        # Formulate spatial score: Sum of all logits for the requested class
        # (e.g. asking "Why did you segment Microaneurysms across this image?")
        score = logits[:, target_class_idx, :, :].sum()
        
        # Backward pass triggers the hook to capture self.gradients
        score.backward(retain_graph=True)
        
        # Compute Grad-CAM Importance Weights (Global Average Pooling on H, W)
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # Weighted combination of feature maps
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        
        # ReLU to keep only features with a POSITIVE influence on the target class
        cam = F.relu(cam)
        
        return cam, logits

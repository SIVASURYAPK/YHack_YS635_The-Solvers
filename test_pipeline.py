import sys
import torch
import numpy as np
from PIL import Image, ImageDraw
import traceback

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None

from interface import get_unet_gradcam

def create_dummy_fundus(size=(512, 512)):
    # Creates a synthetic eye fundus to test the pipeline
    img = Image.new('RGB', size, (0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Retinal background
    draw.ellipse([(20, 20), (492, 492)], fill=(120, 30, 10))
    # Optic disc
    draw.ellipse([(380, 220), (430, 270)], fill=(255, 255, 150))
    return img

def run_validation():
    try:
        print("--- STARTING END-TO-END VALIDATION ---")
        print("[1/4] Loading model checkpoint...")
        data = torch.load('idrid_finetuned.pth', map_location='cpu')
        
        if isinstance(data, torch.nn.Module):
            model = data
            print("      Model object loaded directly!")
        else:
            print("      Checkpoint is a state_dict. Rebuilding SMP U-Net SE-ResNet50...")
            if smp is None:
                print("      ERROR: 'segmentation_models_pytorch' is not installed.")
                print("      Please run: pip install segmentation-models-pytorch")
                return
            
            model = smp.Unet(encoder_name='se_resnet50', encoder_weights=None, classes=5, in_channels=3)
            sd = data.get('state_dict', data.get('model_state_dict', data)) if isinstance(data, dict) else data
            
            # Handle potential DataParallel 'module.' prefix
            if all(k.startswith('module.') for k in sd.keys()):
                sd = {k.replace('module.', ''): v for k, v in sd.items()}
                
            model.load_state_dict(sd, strict=False)
            
        model.eval()
        
        print("[2/4] Generating synthetic fundus test image...")
        img = create_dummy_fundus()
        
        print("[3/4] Passing through Grad-CAM Integration Interface...")
        # Testing channel 1 (typically Microaneurysms or Hemorrhages)
        result = get_unet_gradcam(model, img, target_class_idx=1, target_size=(512, 512))
        
        print("[4/4] Saving output files...")
        result['pil'].save('test_overlay_success.jpg')
        
        print("\n=== VALIDATION SUCCESSFUL! ===")
        print(f"File created: test_overlay_success.jpg")
        print(f"Base64 snippet generated successfully (length: {len(result['base64'])} characters)")
        
    except Exception as e:
        print("\n=== VALIDATION FAILED ===")
        traceback.print_exc()

if __name__ == "__main__":
    run_validation()

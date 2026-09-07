import os
import sys
import glob
import cv2
import matplotlib.pyplot as plt
from enhancer import process_fundus_image

def find_image_path(query):
    """Recursively resolves path if only a filename is passed."""
    if os.path.exists(query):
        return query
    
    filename = os.path.basename(query)
    matches = glob.glob(f"**/{filename}", recursive=True)
    return matches[0] if matches else None

def show_single_demo(image_path):
    filename = os.path.basename(image_path)
    orig, enhanced, status, reasons, metrics = process_fundus_image(image_path)

    if orig is None:
        print(f"Error processing image: {image_path}")
        return

    # 1. Save ONLY the enhanced image to output_images (if status is Mid and not already saved)
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)
    enhanced_save_path = os.path.join(output_dir, f"enhanced_{filename}")

    if status == "Mid" and enhanced is not None:
        if not os.path.exists(enhanced_save_path):
            cv2.imwrite(enhanced_save_path, enhanced)
            print(f"Saved enhanced image to: {enhanced_save_path}")
        else:
            print(f"Enhanced image already exists at {enhanced_save_path}. Skipping save.")

    # 2. Render comparison GUI visualization
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    b = metrics["before"]

    if status == "Ungradeable":
        plt.figure(figsize=(8, 8))
        plt.imshow(orig_rgb, interpolation='bicubic')
        
        metric_str = f"Focus: {b['Focus']:.1f} | Illumination Score: {b['Illumination']:.1f} | FOV: {b['FOV']:.1f}%"
        reason_text = "\n".join([f"• {r}" for r in reasons])
        
        plt.title(f"FILE: {filename}\nSTATUS: UNGRADEABLE (Decision Score: {b['DecisionScore']:.1f}/100)\n\nRECAPTURE ALERT ISSUED:\n{reason_text}\n\n{metric_str}", 
                  color='red', fontsize=11, fontweight='bold')
        plt.axis("off")

    elif status == "Good":
        plt.figure(figsize=(8, 8))
        plt.imshow(orig_rgb, interpolation='bicubic')
        
        metric_str = f"Focus: {b['Focus']:.1f} | Illumination Score: {b['Illumination']:.1f} | FOV: {b['FOV']:.1f}%"
        
        plt.title(f"FILE: {filename}\nSTATUS: GOOD / OPTIMAL (Decision Score: {b['DecisionScore']:.1f}/100)\nAction: No enhancement needed. Passing original image directly to U-Net Pipeline.\n\n{metric_str}", 
                  color='green', fontsize=11, fontweight='bold')
        plt.axis("off")

    elif status == "Mid":
        display_enh = enhanced if enhanced is not None else orig
        enhanced_rgb = cv2.cvtColor(display_enh, cv2.COLOR_BGR2RGB)
        a = metrics["after"] if metrics.get("after") else b

        plt.figure(figsize=(14, 7))
        
        # Subplot 1: Original Image
        plt.subplot(1, 2, 1)
        plt.imshow(orig_rgb, interpolation='bicubic')
        orig_title = (f"BEFORE: {filename} (MID Quality)\n"
                      f"Decision Score: {b['DecisionScore']:.1f}/100\n"
                      f"Focus: {b['Focus']:.1f} | Illumination Score: {b['Illumination']:.1f} | FOV: {b['FOV']:.1f}%")
        plt.title(orig_title, fontsize=10)
        plt.axis("off")

        # Subplot 2: Enhanced Image
        plt.subplot(1, 2, 2)
        plt.imshow(enhanced_rgb, interpolation='bicubic')
        enh_title = (f"AFTER: Adaptive LAB-CLAHE Enhanced\n"
                     f"Decision Score: {a['DecisionScore']:.1f}/100\n"
                     f"Focus: {a['Focus']:.1f} | Illumination Score: {a['Illumination']:.1f} | FOV: {a['FOV']:.1f}%")
        plt.title(enh_title, fontsize=10, color='darkgreen')
        plt.axis("off")

    plt.tight_layout()
    
    # 3. Pop up interactive GUI window without calling plt.savefig() for the plot
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        raw_target = sys.argv[1]
        resolved_path = find_image_path(raw_target)
        if resolved_path:
            show_single_demo(resolved_path)
        else:
            print(f"File not found: {raw_target}")
    else:
        dataset_files = glob.glob('resized_train/*.jpeg') + glob.glob('resized_train_cropped/*.jpeg')
        if dataset_files:
            show_single_demo(dataset_files[0])
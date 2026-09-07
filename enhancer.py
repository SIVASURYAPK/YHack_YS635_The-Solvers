import cv2
import numpy as np
import os
import sys
import glob

def extract_fov_mask(img_bgr, threshold=12):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def check_field_of_view(mask):
    total_pixels = mask.size
    retinal_pixels = np.count_nonzero(mask)
    return (retinal_pixels / total_pixels) * 100.0

def compute_quality_metrics(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = extract_fov_mask(img_bgr)
    fov_coverage = check_field_of_view(mask)

    fov_pixels = gray[mask > 0]
    
    if len(fov_pixels) == 0:
        return 0.0, 0.0, 0.0, 0.0, mask

    # 1. Focus Score (Variance of Laplacian)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    focus_score = float(np.var(laplacian[mask > 0]))

    # 2. Illumination Score (Mean Intensity + Histogram Uniformity)
    mean_intensity = float(np.mean(fov_pixels))
    # Normalized entropy of pixel distribution within FOV
    hist, _ = np.histogram(fov_pixels, bins=256, range=(0, 256))
    hist_prob = hist / (hist.sum() + 1e-7)
    hist_prob = hist_prob[hist_prob > 0]
    entropy = -np.sum(hist_prob * np.log2(hist_prob))
    uniformity = entropy / 8.0  # Normalized to 0-1
    
    # Combined Illumination Score (0 - 100 scale)
    intensity_factor = 1.0 - abs(mean_intensity - 128.0) / 128.0
    illumination_score = float(max(0.0, min(100.0, (intensity_factor * 0.6 + uniformity * 0.4) * 100.0)))

    # 3. Overall Decision Logic Score
    # Weighted composite score: Focus (40%), Illumination (40%), FOV (20%)
    norm_focus = min(1.0, focus_score / 150.0)
    norm_illum = illumination_score / 100.0
    norm_fov = min(1.0, fov_coverage / 50.0)
    decision_score = float((norm_focus * 0.4 + norm_illum * 0.4 + norm_fov * 0.2) * 100.0)

    return focus_score, illumination_score, decision_score, fov_coverage, mask

def apply_lab_clahe_enhancement(img_bgr, quality_tier="mid"):
    clip_limit = 2.0 if quality_tier == "poor" else 1.5
    denoise_h = 3 if quality_tier == "poor" else 2
    unsharp_weight = 0.25 if quality_tier == "poor" else 0.15

    denoised = cv2.fastNlMeansDenoisingColored(img_bgr, None, h=denoise_h, hColor=denoise_h, templateWindowSize=7, searchWindowSize=21)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    h_img, w_img = img_bgr.shape[:2]
    sigma_val = max(h_img, w_img) / 16.0
    bg_illum = cv2.GaussianBlur(l, (0, 0), sigmaX=sigma_val, sigmaY=sigma_val)
    
    l_float = l.astype(np.float32) - bg_illum.astype(np.float32)
    l_norm = cv2.normalize(l_float, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_norm)

    l_blur = cv2.GaussianBlur(l_enhanced, (0, 0), sigmaX=1.5)
    l_sharp = cv2.addWeighted(l_enhanced, 1.0 + unsharp_weight, l_blur, -unsharp_weight, 0)

    enhanced_lab = cv2.merge((l_sharp, a, b))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    mask = extract_fov_mask(img_bgr)
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
    return (enhanced_bgr * mask_3ch).astype(np.uint8)

def process_fundus_image(image_path, output_dir="output_images"):
    filename = os.path.basename(image_path)
    
    if not os.path.exists(image_path):
        return None, None, "Error", [f"File not found: {image_path}"], {}

    img = cv2.imread(image_path)
    if img is None:
        return None, None, "Error", [f"Corrupted or unsupported format: {filename}"], {}

    focus, illum, decision_score, fov, mask = compute_quality_metrics(img)
    
    before_metrics = {
        "Focus": focus,
        "Illumination": illum,
        "DecisionScore": decision_score,
        "FOV": fov
    }

    reasons = []
    if focus < 35.0:
        reasons.append(f"Out of focus (Focus score {focus:.1f} < 35.0)")
    if illum < 25.0:
        reasons.append(f"Poor illumination profile ({illum:.1f} < 25.0)")
    if fov < 35.0:
        reasons.append(f"Insufficient Retinal FOV ({fov:.1f}% < 35.0%)")

    if len(reasons) > 0:
        return img, None, "Ungradeable", reasons, {"before": before_metrics, "after": None}

    IS_GOOD = (focus >= 120.0) and (illum >= 60.0) and (decision_score >= 70.0)

    if IS_GOOD:
        return img, img, "Good", [], {"before": before_metrics, "after": before_metrics}

    quality_tier = "poor" if (focus < 60.0 or illum < 40.0) else "mid"
    enhanced_img = apply_lab_clahe_enhancement(img, quality_tier=quality_tier)

    after_focus, after_illum, after_decision_score, after_fov, _ = compute_quality_metrics(enhanced_img)
    after_metrics = {
        "Focus": after_focus,
        "Illumination": after_illum,
        "DecisionScore": after_decision_score,
        "FOV": after_fov
    }

    if after_illum < (illum * 0.8) or after_fov < (fov * 0.8):
        return img, img, "Good", ["Enhancement degraded quality; returning original."], {"before": before_metrics, "after": before_metrics}

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"enhanced_{filename}")
    cv2.imwrite(save_path, enhanced_img)

    return img, enhanced_img, "Mid", [], {"before": before_metrics, "after": after_metrics}

def resolve_image_path(query):
    if os.path.exists(query):
        return query
    filename = os.path.basename(query)
    matches = glob.glob(f"**/{filename}", recursive=True)
    return matches[0] if matches else None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = sys.argv[1]
        img_path = resolve_image_path(target)
        if not img_path:
            print(f"File not found in workspace: {target}")
            sys.exit(1)
        files = [img_path]
    else:
        files = glob.glob("resized_train/*.jpeg") + glob.glob("resized_train_cropped/*.jpeg")
        files = files[:3]

    if not files:
        print("No clinical fundus images found in workspace.")
        sys.exit(1)

    os.makedirs("output_images", exist_ok=True)

    for path in files:
        filename = os.path.basename(path)
        orig, enh, status, reasons, metrics = process_fundus_image(path)
        b = metrics["before"]
        
        print("==================================================")
        print(f"FILE: {filename}")
        print(f"METRICS: Focus={b['Focus']:.1f} | Illumination Score={b['Illumination']:.1f} | FOV={b['FOV']:.1f}%")
        print(f"STATUS : [{status.upper()}] (Decision Logic Score: {b['DecisionScore']:.1f}/100)")
        
        if status == "Ungradeable":
            reason_str = ", ".join(reasons) if reasons else "Quality threshold failure"
            print(f"ACTION : [RECAPTURE ALERT] Rejection Reasons: {reason_str}")
        elif status == "Good":
            print("ACTION : No enhancement needed. Passing original image directly to U-Net Pipeline.")
        elif status == "Mid":
            out_path = os.path.join("output_images", f"enhanced_{filename}")
            print("ACTION : Applying Denoising + Gamma Normalization + LAB-CLAHE Enhancement...")
            print(f"RESULT : Enhanced image saved to {out_path} -> Passing to U-Net Pipeline")
        print("==================================================\n")
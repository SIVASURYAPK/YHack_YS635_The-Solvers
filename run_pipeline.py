import glob
import os
import sys
from multi_disease_classifier import MultiDiseasePipeline

def resolve_image_path(query_path: str = None) -> str:
    if query_path and os.path.exists(query_path):
        return query_path
    
    if query_path:
        filename = os.path.basename(query_path)
        matches = glob.glob(f"**/{filename}", recursive=True)
        if matches:
            return matches[0]

    search_patterns = [
        "resized_train/*.jpeg",
        "resized_train_cropped/*.jpeg",
        "output_images/*.jpeg",
        "*.jpeg",
        "*.png"
    ]
    
    for pattern in search_patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
            
    return None

def run_diagnosis(image_path: str):
    pipeline = MultiDiseasePipeline()
    results = pipeline.process_patient_image(image_path)
    
    qual = results["image_quality"]
    metrics = qual["metrics"]
    report = results["diagnostic_report"]
    enhanced_path = results.get("enhanced_image_path")

    print("\n" + "=" * 65)
    print("           OCULAR MULTI-DISEASE SCREENING REPORT          ")
    print("=" * 65)
    print(f" Image Processed   : {image_path}")
    print(f" Quality Status    : {qual['status'].upper()}")
    print(f" Quality Score     : {qual['decision_score']:.2f} / 100")
    
    if enhanced_path:
        print(f" Enhancement Action: APPLIED (LAB-CLAHE + Denoising)")
        print(f" Enhanced Saved To : {enhanced_path}")
    else:
        print(f" Enhancement Action: NONE REQUIRED")

    print("-" * 65)
    print(" IMAGE QUALITY METRICS:")
    print(f"   • Focus Score       : {metrics['focus']:.2f}")
    print(f"   • Illumination Score: {metrics['illumination']:.2f}")
    print(f"   • Retinal FOV       : {metrics['fov']:.2f}%")
    print("-" * 65)

    if isinstance(report, dict):
        print(" DIAGNOSTIC & SEVERITY EVALUATION:")
        print(f" {'Disease Name':<25} | {'Grade':<7} | {'Clinical Stage'}")
        print(" " + "-" * 61)
        
        for disease_key, details in report.items():
            disease_name = disease_key.replace('_', ' ').title()
            grade = details.get('grade', 'N/A')
            label = details.get('label', 'N/A')
            print(f" {disease_name:<25} | {grade:<7} | {label}")
            
        print(" " + "-" * 61)
    else:
        print(f" DIAGNOSTIC STATUS: {report}")

    print("=" * 65 + "\n")

if __name__ == "__main__":
    target_arg = sys.argv[1] if len(sys.argv) > 1 else None
    sample_image = resolve_image_path(target_arg)

    if not sample_image:
        print("[ERROR] No valid clinical fundus images (.jpeg/.png) found in workspace.")
        sys.exit(1)

    run_diagnosis(sample_image)
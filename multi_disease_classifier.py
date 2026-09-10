import os
import cv2
import numpy as np
from enhancer import QualityEnhancer
from models.disease_classifier import OcularDiseaseClassifier

class MultiDiseasePipeline:
    def __init__(self, output_dir="output_images"):
        self.enhancer = QualityEnhancer()
        self.classifier = OcularDiseaseClassifier()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def process_patient_image(self, image_path: str) -> dict:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        quality_assessment = self.enhancer.assess_quality(image)
        enhanced_saved_path = None
        
        # Check if quality requires enhancement
        if quality_assessment["status"] in ["Borderline", "Mid"]:
            filename = os.path.basename(image_path)
            enhanced_filename = f"enhanced_{filename}"
            target_path = os.path.join(self.output_dir, enhanced_filename)

            # Process enhancement
            processed_image = self.enhancer.enhance_image(image)
            
            # Save the enhanced image to output_images
            cv2.imwrite(target_path, processed_image)
            enhanced_saved_path = target_path
        else:
            processed_image = image.copy()

        # Generate diagnostic predictions
        if quality_assessment["status"] in ["Poor", "Ungradeable"]:
            disease_results = "Rejected: Image quality too poor for reliable diagnostic screening."
        else:
            disease_results = self.classifier.analyze_image(processed_image)

        return {
            "image_quality": quality_assessment,
            "enhanced_image_path": enhanced_saved_path,
            "diagnostic_report": disease_results
        }
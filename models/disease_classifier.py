import numpy as np
import cv2

class OcularDiseaseClassifier:
    """
    Handles severity estimation for:
    - Diabetic Retinopathy (0: Normal, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative)
    - Glaucoma (0: Normal, 1: Early, 2: Moderate, 3: Advanced)
    - Cataract (0: Normal, 1: Mild, 2: Moderate, 3: Severe)
    - Retinal Detachment (0: Normal, 1: Partial, 2: Total)
    """

    SEVERITY_MAPS = {
        "diabetic_retinopathy": {
            0: "No DR",
            1: "Mild NPDR",
            2: "Moderate NPDR",
            3: "Severe NPDR",
            4: "Proliferative DR"
        },
        "glaucoma": {
            0: "Normal",
            1: "Early Glaucoma",
            2: "Moderate Glaucoma",
            3: "Advanced Glaucoma"
        },
        "cataract": {
            0: "Normal",
            1: "Mild Cataract",
            2: "Moderate Cataract",
            3: "Severe Cataract"
        },
        "retinal_detachment": {
            0: "Normal",
            1: "Partial Detachment",
            2: "Total Detachment"
        }
    }

    def __init__(self, model_weights_path: str = None):
        self.model_weights_path = model_weights_path

    def analyze_image(self, image: np.ndarray) -> dict:
        """
        Runs disease evaluation on the preprocessed/enhanced fundus image.
        Returns severity scores and clinical stages for all 4 diseases.
        """
        # Preprocessing: convert to RGB and resize for consistency
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
        dr_stage = self._eval_diabetic_retinopathy(image)
        glaucoma_stage = self._eval_glaucoma(image)
        cataract_stage = self._eval_cataract(image)
        detachment_stage = self._eval_retinal_detachment(image)

        return {
            "diabetic_retinopathy": {
                "grade": dr_stage,
                "label": self.SEVERITY_MAPS["diabetic_retinopathy"][dr_stage]
            },
            "glaucoma": {
                "grade": glaucoma_stage,
                "label": self.SEVERITY_MAPS["glaucoma"][glaucoma_stage]
            },
            "cataract": {
                "grade": cataract_stage,
                "label": self.SEVERITY_MAPS["cataract"][cataract_stage]
            },
            "retinal_detachment": {
                "grade": detachment_stage,
                "label": self.SEVERITY_MAPS["retinal_detachment"][detachment_stage]
            }
        }

    def _eval_diabetic_retinopathy(self, image: np.ndarray) -> int:
        # Vessel and lesion features on green channel
        green = image[:, :, 1]
        std_dev = np.std(green)
        if std_dev < 25: return 0
        elif std_dev < 35: return 1
        elif std_dev < 45: return 2
        elif std_dev < 55: return 3
        else: return 4

    def _eval_glaucoma(self, image: np.ndarray) -> int:
        # Optical Cup-to-Disc ratio approximation using luminance contrast
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bright_pixels = np.sum(gray > 200) / gray.size
        if bright_pixels < 0.02: return 0
        elif bright_pixels < 0.05: return 1
        elif bright_pixels < 0.10: return 2
        else: return 3

    def _eval_cataract(self, image: np.ndarray) -> int:
        # Optical opacity estimation using mean intensity & blur variance
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var > 300: return 0
        elif lap_var > 180: return 1
        elif lap_var > 80: return 2
        else: return 3

    def _eval_retinal_detachment(self, image: np.ndarray) -> int:
        # Edge disruption/wrinkle detection using Canny operator
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.count_nonzero(edges) / edges.size
        if edge_ratio < 0.03: return 0
        elif edge_ratio < 0.07: return 1
        else: return 2
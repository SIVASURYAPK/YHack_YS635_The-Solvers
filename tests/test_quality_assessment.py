import unittest
import cv2
import numpy as np
from enhancer import compute_quality_metrics

class TestQualityAssessment(unittest.TestCase):

    def setUp(self):
        # Full circular retinal disk (r=230 covers ~63% of 512x512 canvas)
        self.sharp_img = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.circle(self.sharp_img, (256, 256), 230, (30, 70, 180), -1)
        cv2.line(self.sharp_img, (256, 256), (100, 100), (10, 10, 80), 5)
        
        # Blurred reference canvas
        self.blur_img = cv2.GaussianBlur(self.sharp_img, (25, 25), 8.0)

        # Well-lit canvas
        self.bright_img = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.circle(self.bright_img, (256, 256), 230, (180, 180, 220), -1)

        # Dark canvas
        self.dark_img = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.circle(self.dark_img, (256, 256), 230, (25, 25, 30), -1)

    def test_T1_1_focus_score_sharp(self):
        focus, _, _, _, _ = compute_quality_metrics(self.sharp_img)
        norm_focus = min(1.0, focus / 150.0)
        self.assertGreater(norm_focus, 0.6, f"Measured focus score {norm_focus:.2f} <= 0.6")

    def test_T1_2_focus_score_blur(self):
        focus, _, _, _, _ = compute_quality_metrics(self.blur_img)
        norm_focus = min(1.0, focus / 150.0)
        self.assertLess(norm_focus, 0.5, f"Measured focus score {norm_focus:.2f} >= 0.5")

    def test_T1_3_illumination_score_well_lit(self):
        _, brightness, _, _, _ = compute_quality_metrics(self.bright_img)
        illum_score = brightness / 255.0
        self.assertGreaterEqual(illum_score, 0.60, f"Illumination {illum_score:.2f} < 0.60")

    def test_T1_4_illumination_score_dark(self):
        _, brightness, _, _, _ = compute_quality_metrics(self.dark_img)
        illum_score = brightness / 255.0
        self.assertLess(illum_score, 0.40, f"Illumination {illum_score:.2f} >= 0.40")

    def test_T1_5_fov_score_full(self):
        _, _, _, fov, _ = compute_quality_metrics(self.sharp_img)
        fov_score = fov / 100.0
        self.assertGreater(fov_score, 0.60, f"FOV score {fov_score:.2f} <= 0.60")

    def test_T1_6_fov_score_partial(self):
        partial_img = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.circle(partial_img, (256, 256), 100, (30, 70, 180), -1)
        _, _, _, fov, _ = compute_quality_metrics(partial_img)
        fov_score = fov / 100.0
        self.assertLess(fov_score, 0.50, f"Partial FOV score {fov_score:.2f} >= 0.50")

if __name__ == "__main__":
    unittest.main()
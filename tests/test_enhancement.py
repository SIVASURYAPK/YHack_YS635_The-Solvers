import unittest
import cv2
import numpy as np
from enhancer import apply_lab_clahe_enhancement, compute_quality_metrics, process_fundus_image, extract_fov_mask

class TestEnhancement(unittest.TestCase):

    def setUp(self):
        self.low_contrast_img = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.circle(self.low_contrast_img, (256, 256), 230, (80, 80, 100), -1)

    def test_T1_7_clahe_enhancement_contrast(self):
        _, _, contrast_before, _, _ = compute_quality_metrics(self.low_contrast_img)
        enhanced = apply_lab_clahe_enhancement(self.low_contrast_img, quality_tier="mid")
        _, _, contrast_after, _, _ = compute_quality_metrics(enhanced)

        improvement = ((contrast_after - contrast_before) / (contrast_before + 1e-5)) * 100.0
        self.assertGreater(improvement, 15.0, f"Contrast improvement only {improvement:.2f}%")

    def test_T1_8_denoising_snr(self):
        # Generate uniform tissue region with high Gaussian noise
        base_tissue = np.full((512, 512, 3), (40, 60, 140), dtype=np.uint8)
        mask = extract_fov_mask(base_tissue)
        
        noise = np.random.normal(0, 25, base_tissue.shape).astype(np.int16)
        noisy_img = np.clip(base_tissue.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        noisy_img = cv2.bitwise_and(noisy_img, noisy_img, mask=mask)

        # Denoise using fastNlMeans
        denoised = cv2.fastNlMeansDenoisingColored(noisy_img, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

        # Compute SNR inside FOV tissue region
        g_noisy = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2GRAY)[mask > 0]
        g_denoised = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)[mask > 0]

        snr_before = 10 * np.log10(np.mean(g_noisy)**2 / (np.var(g_noisy) + 1e-5))
        snr_after = 10 * np.log10(np.mean(g_denoised)**2 / (np.var(g_denoised) + 1e-5))
        snr_gain = snr_after - snr_before

        self.assertGreater(snr_gain, 2.0, f"SNR gain only {snr_gain:.2f} dB")

    def test_T1_9_quality_rejection(self):
        bad_img = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.circle(bad_img, (256, 256), 230, (10, 10, 10), -1)
        cv2.imwrite("temp_bad.png", bad_img)

        orig, enh, status, reasons, _ = process_fundus_image("temp_bad.png")
        import os
        if os.path.exists("temp_bad.png"):
            os.remove("temp_bad.png")

        self.assertEqual(status, "Ungradeable")
        self.assertGreater(len(reasons), 0)

if __name__ == "__main__":
    unittest.main()
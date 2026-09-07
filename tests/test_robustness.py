import unittest
import cv2
import numpy as np
import os
from enhancer import process_fundus_image

class TestRobustness(unittest.TestCase):

    def test_corrupted_file_handling(self):
        with open("invalid.png", "w") as f:
            f.write("corrupted content")

        orig, enh, status, reasons, _ = process_fundus_image("invalid.png")
        if os.path.exists("invalid.png"):
            os.remove("invalid.png")

        self.assertEqual(status, "Error")

    def test_extreme_darkness(self):
        dark_img = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.imwrite("temp_dark.png", dark_img)

        _, _, status, _, _ = process_fundus_image("temp_dark.png")
        if os.path.exists("temp_dark.png"):
            os.remove("temp_dark.png")

        self.assertEqual(status, "Ungradeable")

if __name__ == "__main__":
    unittest.main()
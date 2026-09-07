import unittest
import time
import numpy as np
from enhancer import process_fundus_image
import cv2
import os

class TestPerformance(unittest.TestCase):

    def test_T1_10_processing_time(self):
        sample_img = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.circle(sample_img, (256, 256), 200, (30, 70, 180), -1)
        cv2.imwrite("temp_perf.png", sample_img)

        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            process_fundus_image("temp_perf.png")
            t1 = time.perf_counter()
            times.append(t1 - t0)

        if os.path.exists("temp_perf.png"):
            os.remove("temp_perf.png")

        avg_time = np.mean(times)
        max_time = np.max(times)

        self.assertLess(avg_time, 2.0, f"Average execution time {avg_time:.2f}s >= 2.0s")

if __name__ == "__main__":
    unittest.main()
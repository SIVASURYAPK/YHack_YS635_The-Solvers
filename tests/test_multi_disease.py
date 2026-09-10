import unittest
import numpy as np
from models.disease_classifier import OcularDiseaseClassifier

class TestMultiDiseaseClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = OcularDiseaseClassifier()
        self.dummy_image = np.uint8(np.random.rand(224, 224, 3) * 255)

    def test_output_keys(self):
        res = self.classifier.analyze_image(self.dummy_image)
        self.assertIn("diabetic_retinopathy", res)
        self.assertIn("glaucoma", res)
        self.assertIn("cataract", res)
        self.assertIn("retinal_detachment", res)

    def test_severity_grades(self):
        res = self.classifier.analyze_image(self.dummy_image)
        self.assertIn(res["diabetic_retinopathy"]["grade"], range(5))
        self.assertIn(res["glaucoma"]["grade"], range(4))
        self.assertIn(res["cataract"]["grade"], range(4))
        self.assertIn(res["retinal_detachment"]["grade"], range(3))

if __name__ == "__main__":
    unittest.main()
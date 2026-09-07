import unittest
import os
import csv
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from enhancer import process_fundus_image

class TestAccuracy(unittest.TestCase):

    def test_accuracy_metrics(self):
        meta_path = os.path.join("quality_test_set", "metadata.csv")
        if not os.path.exists(meta_path):
            self.skipTest("quality_test_set/metadata.csv not found.")

        y_true = []
        y_pred = []
        y_scores = []

        with open(meta_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fpath = os.path.join("quality_test_set", row['filename'])
                if not os.path.exists(fpath):
                    continue

                gt = 0 if row['ground_truth'] == 'ungradeable' else 1
                y_true.append(gt)

                _, _, status, _, metrics = process_fundus_image(fpath)
                pred = 0 if status == 'Ungradeable' else 1
                y_pred.append(pred)

                score = metrics['before']['Focus'] / 150.0
                y_scores.append(score)

        if len(y_true) == 0:
            self.skipTest("No dataset files loaded.")

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        acc = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        try:
            auc = roc_auc_score(y_true, y_scores)
        except Exception:
            auc = 0.50

        self.assertGreaterEqual(acc, 0.70, f"System accuracy {acc:.2f} < 0.70")

if __name__ == "__main__":
    unittest.main()
import unittest
import numpy as np
import os
import shutil
from src.metrics import save_classification_report, save_confusion_matrix


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.y_true = [0, 1, 2, 2, 1]
        self.y_pred = [0, 0, 2, 2, 1]
        self.output_dir = "tests/test_outputs"
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        # Usuwa cały katalog test_outputs po każdym teście
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_save_classification_report(self):
        path = os.path.join(self.output_dir, "report.txt")
        save_classification_report(self.y_true, self.y_pred, out_path=path)
        self.assertTrue(os.path.exists(path))
        with open(path, "r") as f:
            content = f.read()
        self.assertIn("precision", content)

    def test_save_confusion_matrix(self):
        path = os.path.join(self.output_dir, "conf_matrix.npy")
        save_confusion_matrix(self.y_true, self.y_pred, out_path=path)
        self.assertTrue(os.path.exists(path))
        matrix = np.load(path)
        self.assertEqual(matrix.shape[0], matrix.shape[1])


if __name__ == "__main__":
    unittest.main()

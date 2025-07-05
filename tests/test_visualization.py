import unittest
import numpy as np
import os
from src.visualization import plot_tsne


class TestVisualization(unittest.TestCase):

    def setUp(self):
        self.X_tsne = np.random.rand(20, 2)
        self.labels = np.array([0]*10 + [1]*10)
        self.filename = "tsne_test.png"
        self.full_path = os.path.join("outputs", self.filename)

    def tearDown(self):
        if os.path.exists(self.full_path):
            os.remove(self.full_path)

    def test_plot_tsne_creates_file(self):
        plot_tsne(self.X_tsne, self.labels, title="Test tSNE", filename=self.filename)
        self.assertTrue(os.path.exists(self.full_path), f"Plik {self.full_path} nie został utworzony.")


if __name__ == "__main__":
    unittest.main()

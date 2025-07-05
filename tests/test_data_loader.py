import unittest
import numpy as np
from collections import Counter


class TestDataLoader(unittest.TestCase):
    def test_minimum_classes(self):
        # Symulowane etykiety — 2 klasy
        y = np.array(["001"] * 5 + ["002"] * 5)
        self.assertGreater(len(set(y)), 1, "Znaleziono mniej niż 2 klasy")

    def test_balanced_classes_warning(self):
        # Symulowane etykiety — niezbalansowany zbiór
        y = np.array(["001"] * 9 + ["002"])
        class_counts = Counter(y)
        most_common_ratio = max(class_counts.values()) / len(y)
        self.assertGreater(most_common_ratio, 0.8)


if __name__ == "__main__":
    unittest.main()

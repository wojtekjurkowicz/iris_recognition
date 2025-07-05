import os
import unittest
import numpy as np
from src.segmentation import segment_iris


class TestSegmentation(unittest.TestCase):
    def setUp(self):
        os.makedirs("fallbacks", exist_ok=True)
        # Zapisz stan plików przed testem
        self.initial_fallbacks = set(os.listdir("fallbacks"))

    def tearDown(self):
        # Usuń tylko nowe pliki fallbacków, utworzone w trakcie testu
        current = set(os.listdir("fallbacks"))
        new_files = current - self.initial_fallbacks
        for fname in new_files:
            if fname.endswith(".png"):
                os.remove(os.path.join("fallbacks", fname))

    def test_fallback_written(self):
        before = len([f for f in os.listdir("fallbacks") if f.endswith(".png")])

        dummy = np.zeros((100, 100), dtype=np.uint8)
        _ = segment_iris(dummy, disable_fallback_write=False)

        after = len([f for f in os.listdir("fallbacks") if f.endswith(".png")])
        self.assertEqual(after, before + 1)

    def test_invalid_input(self):
        with self.assertRaises(AssertionError):
            segment_iris(None)
        with self.assertRaises(ValueError):
            segment_iris(np.zeros((100, 100, 3)))  # RGB


if __name__ == "__main__":
    unittest.main()

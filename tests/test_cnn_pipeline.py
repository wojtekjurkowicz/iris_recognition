import unittest
import numpy as np
from src.cnn_pipeline import build_embedding_model, build_classifier_model, IrisDataGenerator
from src.config import IMG_SIZE, BATCH_SIZE


class TestCNNPipeline(unittest.TestCase):

    def test_embedding_model_output_shape(self):
        # Testuje czy embedding model zwraca tensor o wymiarze (1, 128)
        model = build_embedding_model((*IMG_SIZE, 3))
        dummy_input = np.random.rand(1, *IMG_SIZE, 3).astype(np.float32)
        output = model(dummy_input)
        self.assertEqual(output.shape[-1], 128)

    def test_classifier_model_output_classes(self):
        # Sprawdza czy klasyfikator zwraca poprawną liczbę klas
        embedding_model = build_embedding_model((*IMG_SIZE, 3))
        classifier_model = build_classifier_model(embedding_model, num_classes=5)
        dummy_input = np.random.rand(1, *IMG_SIZE, 3).astype(np.float32)
        output = classifier_model(dummy_input)
        self.assertEqual(output.shape[-1], 5)

    def test_data_generator_shape(self):
        # Sprawdza, czy generator danych zwraca batch o odpowiednim kształcie
        X = np.random.rand(10, IMG_SIZE[0], IMG_SIZE[1])
        y = np.eye(5)[np.random.choice(5, 10)]
        gen = IrisDataGenerator(X, y, batch_size=2, augment=False)
        batch_x, batch_y = gen[0]
        self.assertEqual(batch_x.shape, (2, IMG_SIZE[0], IMG_SIZE[1], 3))
        self.assertEqual(batch_y.shape, (2, 5))


if __name__ == "__main__":
    unittest.main()

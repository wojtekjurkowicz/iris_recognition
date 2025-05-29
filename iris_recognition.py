import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os


def preprocess_iris(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    blurred = cv2.medianBlur(img, 5)
    edges = cv2.Canny(blurred, 50, 150)
    return edges


def compare_iris(img1_path, img2_path):
    img1 = preprocess_iris(img1_path)
    img2 = preprocess_iris(img2_path)

    similarity, _ = ssim(img1, img2, full=True)
    return similarity


# Przykładowe zdjęcia tęczówki
base_dir = "iris_samples"  # folder ze zdjęciami
iris_reference = os.path.join(base_dir, "person1_1.jpg")
iris_to_compare = os.path.join(base_dir, "person1_2.jpg")  # ten sam użytkownik
different_person = os.path.join(base_dir, "person2_1.jpg")  # inna osoba

score_same = compare_iris(iris_reference, iris_to_compare)
score_diff = compare_iris(iris_reference, different_person)

print(f"Similarity (same person): {score_same:.4f}")
print(f"Similarity (different person): {score_diff:.4f}")

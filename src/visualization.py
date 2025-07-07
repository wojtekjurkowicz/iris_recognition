import random

import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from src.segmentation import segment_iris
from src.config import IMG_SIZE


def plot_tsne(X_tsne, labels, title="t-SNE", filename="tsne_plot.png"):
    plt.figure(figsize=(10, 6))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        idx = labels == label
        x = X_tsne[idx]
        plt.scatter(x[:, 0], x[:, 1], label=str(label), s=15)
    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(os.path.join("outputs", filename))
    plt.close()


def visualize_pipeline_for_user(dataset_path):
    files = [f for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print("[BŁĄD] Brak obrazów w folderze.")
        return

    fname = random.choice(files)
    class_id = fname[:3]  # zakładamy, że nazwa pliku zaczyna się od np. "005_xxx.png"
    print(f"[INFO] Losowo wybrano klasę: {class_id}")

    img_path = os.path.join(dataset_path, fname)
    print(f"[DEBUG] Ładowanie pliku: {img_path}")

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        print(f"[BŁĄD] Nie można wczytać obrazu: {img_path}")
        return

    segmented = segment_iris(img)

    plt.subplot(1, 2, 1)
    plt.title("Oryginał")
    plt.imshow(img, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Po segmentacji")
    plt.imshow(segmented, cmap='gray')

    plt.suptitle(f"Segmentacja klasy {class_id}")
    plt.tight_layout()
    plt.show()

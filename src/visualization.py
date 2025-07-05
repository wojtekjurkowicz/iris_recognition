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


def visualize_pipeline_for_user(class_id, dataset_path):
    # znajdź pierwsze zdjęcie danej klasy
    for fname in os.listdir(dataset_path):
        if class_id in fname:
            img_path = os.path.join(dataset_path, fname)
            break
    else:
        print(f"Nie znaleziono próbki z klasą {class_id}")
        return

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
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

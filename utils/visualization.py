import matplotlib.pyplot as plt
import numpy as np


def plot_tsne(X_tsne, labels, title="t-SNE", filename="tsne_plot.png"):
    plt.figure(figsize=(10, 6))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        idx = labels == label
        plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=str(label), s=15)
    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

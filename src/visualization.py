import random
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from src.segmentation import segment_iris
from src.config import IMG_SIZE
import pandas as pd
import seaborn as sns


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
    people_folders = sorted(os.listdir(dataset_path))
    user_folder = random.choice(people_folders)
    user_path = os.path.join(dataset_path, user_folder)
    print(f"[WIZUALIZACJA] Wybrany użytkownik: {user_folder}")

    # Wybierz losowy obrazek tego użytkownika
    image_files = [f for f in os.listdir(user_path) if f.lower().endswith(('.jpg', '.png', '.bmp'))]
    if not image_files:
        print("Brak obrazów dla użytkownika.")
        return
    chosen_file = random.choice(image_files)
    image_path = os.path.join(user_path, chosen_file)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Błąd wczytywania obrazu: {image_path}")
        return

    # Ścieżka debug zapisu wykrycia
    debug_img_path = os.path.join("outputs", f"debug_user_{user_folder}.png")
    segmented = segment_iris(img, debug_path=debug_img_path)

    # Wyświetl porównanie
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Oryginalny obraz")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Po segmentacji (tęczówka)")
    plt.imshow(segmented, cmap='gray')
    plt.axis('off')

    # Dodaj debugowy obraz z okręgiem
    if os.path.exists(debug_img_path):
        debug_img = cv2.imread(debug_img_path)
        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 3, 3)
        plt.title("Debug (wykryty okrąg)")
        plt.imshow(debug_img)
        plt.axis('off')

    plt.suptitle(f"Użytkownik: {user_folder} | Plik: {chosen_file}")
    os.makedirs("outputs", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"outputs/user_{user_folder}.png")
    plt.close()


def plot_training_metrics(log_path="training_log.csv"):
    if not os.path.exists(log_path):
        print(f"[WARN] Nie znaleziono {log_path}")
        return

    df = pd.read_csv(log_path)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(df["accuracy"], label="Train Acc")
    plt.plot(df["val_accuracy"], label="Val Acc")
    plt.title("Dokładność")
    plt.xlabel("Epoka")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(df["loss"], label="Train Loss")
    plt.plot(df["val_loss"], label="Val Loss")
    plt.title("Strata")
    plt.xlabel("Epoka")
    plt.ylabel("Loss")
    plt.legend()

    os.makedirs("outputs", exist_ok=True)
    plt.tight_layout()
    plt.savefig("outputs/training_metrics.png")
    plt.close()


def plot_confusion_matrix(cm_path="outputs/confusion_matrix.npy"):
    if not os.path.exists(cm_path):
        print(f"[WARN] Nie znaleziono {cm_path}")
        return

    cm = np.load(cm_path)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues", square=True, cbar=True)
    plt.title("Macierz Pomyłek")
    plt.xlabel("Predykcja")
    plt.ylabel("Rzeczywista")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix_plot.png")
    plt.close()


def visualize_fallback_samples(n=8):
    fallback_dir = "fallbacks"
    if not os.path.exists(fallback_dir):
        print("[WARN] Brak folderu fallbacks/")
        return
    files = [f for f in os.listdir(fallback_dir) if f.endswith(".png")]
    if not files:
        print("[INFO] Brak fallbacków do wizualizacji.")
        return

    chosen = random.sample(files, min(n, len(files)))
    plt.figure(figsize=(15, 5))
    for i, fname in enumerate(chosen):
        img = cv2.imread(os.path.join(fallback_dir, fname), cv2.IMREAD_GRAYSCALE)
        plt.subplot(1, n, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(fname)
        plt.axis("off")
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/fallback_samples.png")
    plt.close()
    print(f"[INFO] Zapisano fallback_samples.png")


def analyze_fallback_patterns(dataset_path=""):
    fallback_dir = "fallbacks"
    if not os.path.exists(fallback_dir):
        return

    user_counts = {}
    for fname in os.listdir(fallback_dir):
        if "_" in fname:
            uid = fname.split("_")[1].split(".")[0]
            user_counts[uid] = user_counts.get(uid, 0) + 1

    if user_counts:
        sorted_counts = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)
        print("[ANALIZA] Najczęstsze fallbacki dla użytkowników:")
        for uid, count in sorted_counts[:10]:
            print(f"Użytkownik {uid}: {count} fallbacków")

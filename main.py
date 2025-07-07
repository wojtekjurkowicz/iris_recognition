import argparse
import os
import time
import tensorflow as tf
from tensorflow.sysconfig import get_build_info as tf_build_info
from src.data_loader import load_or_segment_data
from src.config import DATASET_PATH
from keras import config
from src.cnn_pipeline import run_cnn
from collections import Counter
from src.visualization import visualize_pipeline_for_user
config.enable_unsafe_deserialization()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=int, default=None, help="Use only a subset of data")
    parser.add_argument('--no-viz', action='store_true', help="Nie pokazuj wizualizacji pipeline'u")
    parser.add_argument('--epochs', type=int, default=None, help="Liczba epok treningu")
    parser.add_argument('--batch', type=int, default=None, help="Wielkość batcha")
    args = parser.parse_args()

    start = time.time()

    X, y = load_or_segment_data(DATASET_PATH, use_hog=False)

    if args.subset is not None:
        print(f"Using subset: {args.subset} samples")
        if args.subset < len(X):
            X = X[:args.subset]
            y = y[:args.subset]
        else:
            print(f"Warning: requested subset {args.subset} > available {len(X)}. Using all samples.")

    # Walidacja liczby klas
    unique_classes = set(y)
    if len(unique_classes) < 2:
        raise ValueError("Znaleziono mniej niż 2 klasy. Nie można trenować klasyfikatora.")

    # ⚖Sprawdzenie niezbalansowanych danych
    class_counts = Counter(y)
    most_common_ratio = max(class_counts.values()) / len(y)
    if most_common_ratio > 0.8:
        print("[UWAGA] Dane są mocno niezbalansowane (jedna klasa >80%).")

    run_cnn(X, y, epochs=args.epochs, batch_size=args.batch)

    if os.path.exists("fallbacks"):
        fallback_count = len([f for f in os.listdir("fallbacks") if f.endswith(".png")])
        print(f"Fallback użyty w segmentacji: {fallback_count} razy")

    print(f"Liczba unikalnych klas: {len(unique_classes)}")
    print(class_counts)

    if not args.no_viz:
        visualize_pipeline_for_user("005", DATASET_PATH)

    end = time.time()
    print(f"Czas wykonania: {end - start:.2f} sekund")


if __name__ == "__main__":
    import sys
    print("GPU available", tf.config.list_physical_devices('GPU'))
    print(tf.__version__)
    print(tf_build_info())
    sys.argv = ["program", "--subset", "10000"]
    main()

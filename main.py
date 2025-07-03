import argparse
import time
from data_loader import load_or_segment_data
from config import DATASET_PATH
from keras import config
from svm_pipeline import run_svm
from cnn_pipeline import run_cnn
from collections import Counter
from segmentation import get_fallback_count
config.enable_unsafe_deserialization()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['svm', 'cnn'], required=True, help="Model to run: svm or cnn")
    parser.add_argument('--subset', type=int, default=None, help="Use only a subset of data")
    parser.add_argument('--classifier', choices=['softmax', 'svm'], default='softmax')
    args = parser.parse_args()

    start = time.time()

    if args.model == 'svm':
        X, y = load_or_segment_data(DATASET_PATH, use_hog=True)
        run_svm(X, y)
    elif args.model == 'cnn':
        X, y = load_or_segment_data(DATASET_PATH, use_hog=False)
        run_cnn(X, y, classifier=args.classifier)

    num_classes = len(set(y))
    print(f"Liczba unikalnych klas: {num_classes}")

    if args.subset is not None:
        min_samples_per_class = 5  # można to uczynić argumentem
        min_required = num_classes * min_samples_per_class

        if args.subset < min_required:
            print(f"[INFO] Podany subset={args.subset} jest zbyt mały względem liczby klas.")
            print(f"[INFO] Zwiększam subset do {min_required} (czyli {min_samples_per_class} próbek na klasę).")
            args.subset = min_required

        print(f"Using subset: {args.subset} samples")
        if args.subset < len(X):
            X = X[:args.subset]
            y = y[:args.subset]
        else:
            print(f"Warning: requested subset {args.subset} > available {len(X)}. Using all samples.")

    print(Counter(y))

    print(f"Fallback użyty w segmentacji: {get_fallback_count()} razy")

    end = time.time()
    print(f"Czas wykonania: {end - start:.2f} sekund")


if __name__ == "__main__":
    import sys

    sys.argv = ["program", "--model", "cnn", "--subset", "5000", "--classifier", "softmax"]
    main()

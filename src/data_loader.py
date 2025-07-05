import os
from glob import glob
import cv2
import numpy as np
from skimage.feature import hog
from src.segmentation import segment_iris


def load_data(base_path, use_hog=True):
    print(f"Loading from: {base_path}")
    X, y = [], []
    people = sorted(os.listdir(base_path))  # All folders (000 to 999)
    # people = sorted(os.listdir(base_path))[:5]
    print(f"Found {len(people)} people folders")
    for person in people:
        person_path = os.path.join(base_path, person)
        if not os.path.isdir(person_path):
            continue
        for eye in ["L", "R"]:
            eye_path = os.path.join(person_path, eye)
            if not os.path.exists(eye_path):
                continue
            for img_file in glob(os.path.join(eye_path, "*.jpg")):
                img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = segment_iris(img)
                if use_hog:
                    features = hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False,
                                   block_norm='L2-Hys')
                    X.append(features)
                else:
                    X.append(img)
                y.append(person)
    print(f"Total loaded samples: {len(X)}")
    return np.array(X), np.array(y)


def load_or_segment_data(dataset_path, use_hog=False):
    cache_path = f"segmented_data_hog{int(use_hog)}.npz"
    if os.path.exists(cache_path):
        print(f"Loading cached segmented data from: {cache_path}")
        data = np.load(cache_path)
        return data["X"], data["y"]
    else:
        print("No cached data found, segmenting from scratch...")
        X, y = load_data(dataset_path, use_hog=use_hog)
        np.savez_compressed(cache_path, X=X, y=y)
        return X, y
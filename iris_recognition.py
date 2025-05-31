import os
import cv2
import time
import argparse
import numpy as np
from glob import glob
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from skimage.feature import hog
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from keras.applications.resnet50 import preprocess_input
import kagglehub

IMG_SIZE = (224, 224)
BATCH_SIZE = 128
EPOCHS = 50

path = kagglehub.dataset_download("sondosaabed/casia-iris-thousand")
print("Path to dataset files:", path)
DATASET_PATH = os.path.join(path, "CASIA-Iris-Thousand", "CASIA-Iris-Thousand")


def load_data(base_path, use_hog=True):
    print(f"Loading from: {base_path}")
    X, y = [], []
    people = sorted(os.listdir(base_path))  # All folders (000 to 999)
    #people = sorted(os.listdir(base_path))[:100]
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
                img = cv2.resize(img, IMG_SIZE)
                if use_hog:
                    features = hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=False)
                    X.append(features)
                else:
                    X.append(img)
                y.append(person)
    print(f"Total loaded samples: {len(X)}")
    return np.array(X), np.array(y)


def run_svm(X, y):
    print("Running SVM classifier...")
    pca = PCA(n_components=100)
    X_pca = pca.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, stratify=y, random_state=42)
    clf = SVC(kernel='linear', C=50)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))


def run_cnn(X, y):
    print("Running CNN classifier...")
    X = preprocess_input(X.astype("float32"))
    X = np.repeat(X[..., np.newaxis], 3, axis=-1)  # ResNet50: 3 kanały

    classes = sorted(list(set(y)))
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    y_encoded = np.array([label_to_idx[label] for label in y])

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded,
                                                        random_state=42)

    AUTOTUNE = tf.data.AUTOTUNE
    random_rot = tf.keras.layers.RandomRotation(0.1)

    def preprocess(img, label):
        img = random_rot(img)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.1)
        return img, label

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = (
        train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
        .cache()
        .shuffle(2048)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    base_model = tf.keras.applications.ResNet50(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = True
    for layer in base_model.layers[:-50]:  # zamroź tylko najwcześniejsze warstwy
        layer.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(classes), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_encoded), y=y_encoded)
    class_weights = dict(enumerate(class_weights))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2),
        ModelCheckpoint("best_model.keras", monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    loss, acc = model.evaluate(test_ds)
    print(f"Test accuracy: {acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['svm', 'cnn'], required=True, help="Model type: 'svm' or 'cnn'")
    args = parser.parse_args()

    start = time.time()

    print(tf.config.list_physical_devices('GPU'))

    if args.model == 'svm':
        X, y = load_data(DATASET_PATH, use_hog=True)
        run_svm(X, y)
    elif args.model == 'cnn':
        X, y = load_data(DATASET_PATH, use_hog=False)
        run_cnn(X, y)

    end = time.time()
    print(f"Czas wykonania: {end - start:.2f} sekund")


if __name__ == "__main__":
    main()

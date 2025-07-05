# from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers, models
from keras import regularizers
from keras.applications.efficientnet import EfficientNetB0
from keras.applications.efficientnet import preprocess_input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
from config import IMG_SIZE, BATCH_SIZE, EPOCHS
from segmentation import segment_iris

matplotlib.use("Agg")  # lub "TkAgg" jeśli masz GUI


def visualize_pipeline_for_user(user_id, dataset_path):
    """
    Wyświetla kolejne etapy przetwarzania jednego oka danego użytkownika.
    """
    import os
    from glob import glob

    user_path = os.path.join(dataset_path, user_id, "L")
    img_files = glob(os.path.join(user_path, "*.jpg"))
    if not img_files:
        print(f"Brak zdjęć dla użytkownika {user_id} w {user_path}")
        return

    img = cv2.imread(img_files[0], cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Nie udało się wczytać obrazu.")
        return

    # Segmentacja
    segmented = segment_iris(img)

    # Przygotowanie do sieci
    img_3ch = np.repeat(segmented[..., np.newaxis], 3, axis=-1)
    img_preprocessed = preprocess_input(img_3ch.astype("float32"))

    # Wizualizacja
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title("Oryginalne zdjęcie")
    axs[1].imshow(segmented, cmap='gray')
    axs[1].set_title("Po segmentacji (tęczówka)")
    axs[2].imshow(img_preprocessed.astype("float32") / 255.0)
    axs[2].set_title("Wejście do sieci (3 kanały)")

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


class L2Normalization(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)


class IrisDataGenerator(keras.utils.Sequence):
    def __init__(self, X, y, batch_size, augment=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.indices = np.arange(len(X))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_ids = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.X[batch_ids]
        batch_y = self.y[batch_ids]

        batch_x = np.repeat(batch_x[..., np.newaxis], 3, axis=-1)
        batch_x = batch_x.astype("float32")

        if self.augment:
            batch_x = tf.image.resize_with_crop_or_pad(batch_x, IMG_SIZE[0] + 20, IMG_SIZE[1] + 20)
            batch_x = tf.image.random_crop(batch_x, size=(len(batch_x), *IMG_SIZE, 3))
            batch_x = tf.image.random_flip_left_right(batch_x)
            batch_x = tf.image.random_brightness(batch_x, max_delta=0.1)
            batch_x = tf.image.random_contrast(batch_x, 0.9, 1.1)
            batch_x = tf.clip_by_value(batch_x, 0.0, 255.0)

        return preprocess_input(batch_x), batch_y

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


def build_embedding_model(input_shape):
    base_model = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    x = base_model.output
    # x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.02))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    embedding = layers.Dense(128, activation='relu', name="embedding")(x)

    return models.Model(inputs=base_model.input, outputs=embedding, name="embedding_model")


def build_classifier_model(embedding_model, num_classes):
    inputs = embedding_model.input
    x = embedding_model.output
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=inputs, outputs=outputs, name="classifier_model")


def run_cnn(X, y):
    print(f"Running CNN classifier with softmax head...")
    print("Shape przed:", X.shape)

    if X.shape[-1] != IMG_SIZE[0]:  # dane spłaszczone
        try:
            X = X.reshape(-1, IMG_SIZE[0], IMG_SIZE[1])
        except Exception as e:
            raise ValueError(f"Błąd podczas reshape: {e}")

    classes = sorted(list(set(y)))
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    y_encoded = tf.keras.utils.to_categorical(
        [label_to_idx[label] for label in y],
        num_classes=len(classes)
    )
    print(f"[DEBUG] y_encoded shape: {y_encoded.shape}, num_classes: {len(classes)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        raise ValueError("Zbiór treningowy lub testowy ma 0 próbek — sprawdź subset albo dane wejściowe.")

    print(f"[DEBUG] X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"[DEBUG] X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    train_gen = IrisDataGenerator(X_train, y_train, batch_size=BATCH_SIZE, augment=True)
    test_gen = IrisDataGenerator(X_test, y_test, batch_size=BATCH_SIZE, augment=False)

    assert len(train_gen) > 0, "train_gen jest pusty"
    assert len(test_gen) > 0, "test_gen jest pusty"

    checkpoint_path = f"best_model_softmax.keras"

    if os.path.exists(checkpoint_path):
        print(f"[INFO] Loading model from checkpoint: {checkpoint_path}")
        model = models.load_model(checkpoint_path)
        embedding_model = keras.Model(inputs=model.input, outputs=model.get_layer("embedding").output)

        initial_epoch = 100
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
            ModelCheckpoint(f"best_model_softmax.keras", monitor='val_accuracy', save_best_only=True)
        ]
        history = model.fit(
            train_gen,
            validation_data=test_gen,
            epochs=100,  # możesz też ustawić np. 150
            initial_epoch=initial_epoch,
            callbacks=callbacks
        )
        loss, acc = model.evaluate(test_gen)
        print(f"[SOFTMAX] Test accuracy: {acc:.4f}")
    else:
        embedding_model = build_embedding_model((*IMG_SIZE, 3))
        model = build_classifier_model(embedding_model, num_classes=len(classes))

        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

        y_labels = np.argmax(y_encoded, axis=1)
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_labels), y=y_labels)
        class_weights = dict(enumerate(class_weights))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
            ModelCheckpoint("best_softmax_model.keras", monitor='val_accuracy', save_best_only=True)
        ]
        assert len(train_gen) > 0, "train_gen is empty"
        assert len(test_gen) > 0, "test_gen is empty"
        model.fit(train_gen, validation_data=test_gen, epochs=EPOCHS, callbacks=callbacks,
                  class_weight=class_weights, verbose=1)

        loss, acc = model.evaluate(test_gen)
        print(f"[SOFTMAX] Test accuracy: {acc:.4f}")

        y_true = np.argmax(np.vstack([y for _, y in test_gen]), axis=1)
        y_pred = np.argmax(model.predict(test_gen), axis=1)

        print("[SOFTMAX] Classification report:")
        print(classification_report(y_true, y_pred))

        with open("report.txt", "w") as f:
            f.write(classification_report(y_true, y_pred))

        cm = confusion_matrix(y_true, y_pred)
        np.save("confusion_matrix.npy", cm)

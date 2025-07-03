# from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.applications.efficientnet import EfficientNetB0, preprocess_input
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from config import IMG_SIZE, BATCH_SIZE, EPOCHS
from keras import layers, models
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import regularizers
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from utils.visualization import plot_tsne
from tensorflow import keras
import random
import matplotlib
matplotlib.use("Agg")  # lub "TkAgg" jeśli masz GUI
import matplotlib.pyplot as plt


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
    embedding = layers.Dense(128, activation='relu')(x)
    embedding = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(embedding)

    return models.Model(inputs=base_model.input, outputs=embedding, name="embedding_model")


def build_classifier_model(embedding_model, num_classes):
    inputs = embedding_model.input
    x = embedding_model.output
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=inputs, outputs=outputs, name="classifier_model")


def run_cnn(X, y, classifier='softmax'):
    print(f"Running CNN classifier with `{classifier}` head...")
    print("Shape przed:", X.shape)

    if X.shape[-1] != IMG_SIZE[0]:  # dane spłaszczone
        try:
            X = X.reshape(-1, IMG_SIZE[0], IMG_SIZE[1])
        except Exception as e:
            raise ValueError(f"Błąd podczas reshape: {e}")

    classes = sorted(list(set(y)))
    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    y_encoded = tf.keras.utils.to_categorical([label_to_idx[label] for label in y])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
    )

    train_gen = IrisDataGenerator(X_train, y_train, batch_size=BATCH_SIZE, augment=True)
    test_gen = IrisDataGenerator(X_test, y_test, batch_size=BATCH_SIZE, augment=False)

    embedding_model = build_embedding_model((*IMG_SIZE, 3))

    if classifier == 'softmax':
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

        model.fit(train_gen, validation_data=test_gen, epochs=EPOCHS, callbacks=callbacks,
                  class_weight=class_weights, verbose=1)

        loss, acc = model.evaluate(test_gen)
        print(f"[SOFTMAX] Test accuracy: {acc:.4f}")

    print("Generating embeddings...")
    X_train_embed = embedding_model.predict(train_gen)
    X_test_embed = embedding_model.predict(test_gen)

    print("Training SVM on embeddings...")
    clf = SVC(kernel='linear', C=10)
    clf.fit(X_train_embed, np.argmax(y_train, axis=1))
    y_pred = clf.predict(X_test_embed)
    print("[SVM on embeddings] Classification report:")
    print(classification_report(np.argmax(y_test, axis=1), y_pred, target_names=classes))

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X_test_embed)
    plot_tsne(X_tsne, y_test, title="Embedding separability (t-SNE)", filename=f"tsne_plot_cnn_{classifier}.png")

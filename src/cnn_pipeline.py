import os
import keras
import matplotlib
import numpy as np
import tensorflow as tf
from keras import models
from keras.applications.efficientnet import preprocess_input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from src.config import IMG_SIZE, BATCH_SIZE, EPOCHS
from src.metrics import save_classification_report, save_confusion_matrix
from src.model_utils import build_embedding_model, build_classifier_model
from src.visualization import plot_training_metrics, plot_confusion_matrix
matplotlib.use("Agg")
csv_logger = CSVLogger('training_log.csv', append=True)


class IrisDataGenerator(keras.utils.Sequence):
    def __init__(self, X, y, batch_size, augment=False, **kwargs):
        super().__init__(**kwargs)
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


def run_cnn(X, y, epochs=None, batch_size=None):
    print(f"Running CNN classifier with softmax head...")
    print("Shape przed:", X.shape)

    batch_size = batch_size or BATCH_SIZE
    epochs = epochs or EPOCHS

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
    train_gen = IrisDataGenerator(X_train, y_train, batch_size=batch_size, augment=True)
    test_gen = IrisDataGenerator(X_test, y_test, batch_size=batch_size, augment=False)

    assert len(train_gen) > 0, "train_gen jest pusty"
    assert len(test_gen) > 0, "test_gen jest pusty"

    os.makedirs("models", exist_ok=True)
    checkpoint_path = "models/best_model_softmax.keras"

    if os.path.exists(checkpoint_path):
        print(f"[INFO] Loading model from checkpoint: {checkpoint_path}")
        model = models.load_model(checkpoint_path)
        embedding_model = keras.Model(inputs=model.input, outputs=model.get_layer("embedding").output)

        initial_epoch = 0
        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
            ModelCheckpoint("models/best_model_softmax.keras", monitor='val_accuracy', save_best_only=True)
        ]
        history = model.fit(
            train_gen,
            validation_data=test_gen,
            epochs=epochs,  # możesz też ustawić np. 150
            initial_epoch=initial_epoch,
            callbacks=callbacks
        )
        loss, acc = model.evaluate(test_gen)
        print(f"[SOFTMAX] Test accuracy: {acc:.4f}")
    else:
        with tf.device('/GPU:0'):
            embedding_model = build_embedding_model((*IMG_SIZE, 3))
            model = build_classifier_model(embedding_model, num_classes=len(classes))

        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

        y_labels = np.argmax(y_encoded, axis=1)
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_labels), y=y_labels)
        class_weights = dict(enumerate(class_weights))

        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
            ModelCheckpoint("models/best_model_softmax.keras", monitor='val_accuracy', save_best_only=True),
            csv_logger
        ]
        assert len(train_gen) > 0, "train_gen is empty"
        assert len(test_gen) > 0, "test_gen is empty"
        import time
        start_train = time.time()
        model.fit(train_gen, validation_data=test_gen, epochs=epochs, callbacks=callbacks,
                  class_weight=class_weights, verbose=1)
        print(f"[TIMER] Training time: {time.time() - start_train:.2f}s")

        start_eval = time.time()
        loss, acc = model.evaluate(test_gen)
        print(f"[TIMER] Evaluation time: {time.time() - start_eval:.2f}s")
        print(f"[SOFTMAX] Test accuracy: {acc:.4f}")

        print("[PREDICTING TEST]: using full batch predict")
        X_all, y_all = [], []
        for X_batch, y_batch in test_gen:
            X_all.append(X_batch)
            y_all.append(y_batch)

        X_all = np.concatenate(X_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)

        start_pred = time.time()
        preds = model.predict(X_all, verbose=1)
        print(f"[TIMER] Prediction time: {time.time() - start_pred:.2f}s")

        y_true = np.argmax(y_all, axis=1)
        y_pred = np.argmax(preds, axis=1)

        print("[SOFTMAX] Classification report:")
        save_classification_report(y_true, y_pred, out_path="outputs/report.txt")
        save_confusion_matrix(y_true, y_pred, out_path="outputs/confusion_matrix.npy")
        plot_training_metrics()
        plot_confusion_matrix()

        from keras import backend as K
        del model
        K.clear_session()

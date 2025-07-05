import tensorflow as tf
from keras import layers, models, regularizers
from keras.applications.efficientnet import EfficientNetB0


class L2Normalization(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)


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

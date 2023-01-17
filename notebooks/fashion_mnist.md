---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
# !pip install Pillow
# !pip install nb_black
```

```{code-cell} ipython3
%load_ext nb_black
```

```{code-cell} ipython3
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

tf.test.is_gpu_available()
```

```{code-cell} ipython3
# TODO get it working quick reading/training with dataset
# from datasets import load_dataset

# hf_ds = load_dataset("fashion_mnist")

# len(hf_ds)  # train, test


# def gen_train():
#     for rec in hf_ds["train"]:
#         yield tf.keras.utils.img_to_array(rec["image"]), rec["label"]


# def gen_test():
#     for rec in hf_ds["test"]:
#         yield tf.keras.utils.img_to_array(rec["image"]), rec["label"]


# def map_method(x, y):
#     return ((x / 255.0) - 0.28) / 0.35, y


# ds_train = (
#     tf.data.Dataset.from_generator(
#         gen_train,
#         output_signature=(
#             tf.TensorSpec(shape=(28, 28, 1), dtype=tf.float32),
#             tf.TensorSpec(shape=(), dtype=tf.uint8),
#         ),
#     )
#     .map(map_method, num_parallel_calls=100)
#     .shuffle(1000)
#     .batch(1024)
#     .prefetch(10)
# )

# ds_test = (
#     tf.data.Dataset.from_generator(
#         gen_test,
#         output_signature=(
#             tf.TensorSpec(shape=(28, 28, 1), dtype=tf.float32),
#             tf.TensorSpec(shape=(), dtype=tf.uint8),
#         ),
#     )
#     .map(map_method, num_parallel_calls=100)
#     .batch(1024)
#     .prefetch(10)
# )

# # look at a single batch
# for x, y in ds_train:
#     break
```

```{code-cell} ipython3
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

x_train = x_train / 255.0
x_train = (x_train - 0.2860405969887956) / 0.3530242445149226

x_test = x_test / 255.0
x_test = (x_test - 0.2860405969887956) / 0.3530242445149226
```

```{code-cell} ipython3
def get_model(learning_rate):
    input_shape = (28, 28, 1)
    num_classes = 10
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(128, kernel_size=3, activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
    )
    return model


def get_model(learning_rate):
    input_shape = (28, 28, 1)
    num_classes = 10
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(
                32, kernel_size=3, activation="relu", padding="same", strides=2
            ),
            layers.Conv2D(
                64, kernel_size=3, activation="relu", padding="same", strides=2
            ),
            layers.Conv2D(
                128, kernel_size=3, activation="relu", padding="same", strides=2
            ),
            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
    )
    return model


def residual_block(x, filters, pooling=False):
    residual = x

    x = layers.Conv2D(filters, 3, use_bias=False, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, 3, use_bias=False, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    if pooling:
        x = layers.MaxPooling2D(2, padding="same")(x)
        residual = layers.Conv2D(filters, 1, strides=2)(residual)
    elif filters != residual.shape[-1]:
        residual = layers.Conv2D(filters, 1)(residual)
    x = layers.add([x, residual])
    return x


def get_model(learning_rate):
    input_shape = (28, 28, 1)
    num_classes = 10
    inputs = keras.Input(shape=input_shape)
    x = residual_block(inputs, filters=32, pooling=True)
    x = residual_block(x, filters=64, pooling=True)
    x = residual_block(x, filters=128, pooling=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
    )
    return model
```

```{code-cell} ipython3
model = get_model(3e-4)
model.summary()
```

```{code-cell} ipython3
class LRFinderCallback(keras.callbacks.Callback):
    def __init__(self, lr_factor=1.1, max_lr=1e-1):
        super().__init__()
        self.factor = lr_factor
        self.lrs = []
        self.losses = []
        self.max_lr = max_lr

    def get_lr(self):
        return float(keras.backend.get_value(self.model.optimizer.learning_rate))

    def set_lr(self, lr):
        keras.backend.set_value(self.model.optimizer.lr, lr)

    def on_train_batch_end(self, batch, logs=None):
        lr = self.get_lr()
        self.lrs.append(lr)
        self.losses.append(logs["loss"])

        if lr > self.max_lr:
            self.model.stop_training = True
            plt.plot(np.log10(self.lrs), self.losses)

        self.set_lr(lr * self.factor)
```

```{code-cell} ipython3
model = get_model(1e-6)
lr_finder = LRFinderCallback(1.1, 1e-1)
model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    batch_size=128,
    epochs=5,
    callbacks=[lr_finder],
    verbose=1,
)
```

```{code-cell} ipython3
model = get_model(3e-3)
model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    batch_size=128,
    epochs=5,
    verbose=1,
)
```

```{code-cell} ipython3
model.layers[0](x)
```

```{code-cell} ipython3
x = x_train[0:32]
x.shape
```

```{code-cell} ipython3
x
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

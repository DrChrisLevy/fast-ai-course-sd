---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
!pip install Pillow
!pip install nb_black
```

```{code-cell} ipython3
%load_ext nb_black
```

```{code-cell} ipython3
import tensorflow as tf

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
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)
```

```{code-cell} ipython3

x_train = x_train / 255.
x_train = (x_train - 0.2860405969887956) / 0.3530242445149226

x_test = x_test / 255.
x_test = (x_test - 0.2860405969887956) / 0.3530242445149226

```

```{code-cell} ipython3
input_shape = (28, 28, 1)
num_classes = 10
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.summary()
opt = keras.optimizers.Adam(learning_rate=3e-3, weight_decay=0.3)
model.compile(
    loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
)
```

```{code-cell} ipython3
model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=128, epochs=5)
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

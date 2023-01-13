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
!pip install datasets
```

```{code-cell} ipython3
%load_ext nb_black
```

```{code-cell} ipython3
import tensorflow as tf
```

```{code-cell} ipython3
from datasets import load_dataset
```

```{code-cell} ipython3
hf_ds = load_dataset("fashion_mnist")
```

```{code-cell} ipython3
len(hf_ds)  # train, test
```

TODO: simplify this and get working with args

```{code-cell} ipython3
def gen_train():
    for rec in hf_ds["train"]:
        yield tf.keras.utils.img_to_array(rec["image"]), rec["label"]


def gen_test():
    for rec in hf_ds["test"]:
        yield tf.keras.utils.img_to_array(rec["image"]), rec["label"]


def map_method(x, y):
    return ((x / 255.0) - 0.28) / 0.35, y
```

```{code-cell} ipython3
ds_train = (
    tf.data.Dataset.from_generator(
        gen_train,
        output_signature=(
            tf.TensorSpec(shape=(28, 28, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.uint8),
        ),
    )
    .map(map_method, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(1000)
    .batch(1024)
    .prefetch(tf.data.AUTOTUNE)
)

ds_test = (
    tf.data.Dataset.from_generator(
        gen_test,
        output_signature=(
            tf.TensorSpec(shape=(28, 28, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.uint8),
        ),
    )
    .map(map_method, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(1024)
    .prefetch(tf.data.AUTOTUNE)
)
```

```{code-cell} ipython3
# look at a single batch
for x, y in ds_train:
    break
```

```{code-cell} ipython3
tf.math.reduce_mean(x)
```

```{code-cell} ipython3
tf.math.reduce_std(x)
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

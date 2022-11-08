---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Learning About Back Prop

+++

## Grab some MNIST Data

```{code-cell} ipython3
from pathlib import Path
import pickle, gzip, math, os, time, shutil, matplotlib as mpl, matplotlib.pyplot as plt

MNIST_URL='https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/data/mnist.pkl.gz?raw=true'
path_data = Path('data')
path_data.mkdir(exist_ok=True)
path_gz = path_data/'mnist.pkl.gz'

from urllib.request import urlretrieve
if not path_gz.exists():
    urlretrieve(MNIST_URL, path_gz)
    
with gzip.open(path_gz, 'rb') as f: 
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
    
x_train, y_train, x_valid, y_valid = map(torch.tensor, [x_train, y_train, x_valid, y_valid])
```

```{code-cell} ipython3
x_train.shape
```

## Simple 2 Layer Network

- TODO: draw a sketch here of the network and activations
- hidden layer with 50 neurons
- $x$ -------> $z_1 = xw_1+b_1$ -------> $a_1 = \text{relu}(z_1)$ -------> $\hat{y} = a_1w_2+b_2$ -------> $\text{loss} = \text{MSE}(y-\hat{y})$

**Tensor Shapes**

$nh=50$ is the number of  neurons in the first layer

$N$ is like the batch dimension. Think of it has $1$ when working through this math! 
Broadcasting can take care of $N$.


$X$ : $(N, 784)$

$w_1$ : $(784, 50)$


$b_1$ : $(N, 50)$

$z_1$ : $(N, 50)$

$a_1$ : $(N, 50)$

$w_2$ : $(50, 1)$

$b_2$ : $(N, 1)$

$\hat{y}$ : $(N, 1)$




```{code-cell} ipython3
x = x_train
n, xdim = x_train.shape
nh = 50 # hidden layer neurons

w1 = torch.rand((xdim, nh))
b1 = torch.zeros((1,nh))
```

```{code-cell} ipython3
(x @ w1).shape, b1.shape
```

```{code-cell} ipython3
z1 = x @ w1 + b1
z1.shape
```

```{code-cell} ipython3
a1 = z1.clamp(min=0.) # relu
a1.shape
```

```{code-cell} ipython3
w2 = torch.rand((nh,1))
b2 = torch.zeros((1,1))
```

```{code-cell} ipython3
w2.shape, b2.shape
```

```{code-cell} ipython3
ypred = a1 @ w2 + b2
ypred.shape
```

```{code-cell} ipython3
ypred.shape, y_train.shape
```

These are not the right shapes for broadcasting when subtracting b/c we will end up with a shape 
(50000,50000).

```{code-cell} ipython3
y_train = y_train[:, None]
y_train.shape
```

Also we want the loss function to be a scalar function so we can compute
its gradient. So we need to reduce the dimension by taking the mean/sum etc.
MSE is not the right loss for this classification problem
but we going to do it to keep things simple!

```{code-cell} ipython3
def loss_func(yp, yh):
    return (yp - yh).square().mean()
```

```{code-cell} ipython3
loss = loss_func(ypred, y_train)
loss
```

```{code-cell} ipython3
def forward_pass(x):
    z1 = x @ w1 + b1
    a1 = z1.clamp(min=0.) # relu
    ypred = a1 @ w2 + b2
    return ypred
```

```{code-cell} ipython3
torch.equal(forward_pass(x_train), ypred)
```

```{code-cell} ipython3
ypred = forward_pass(x_train)
```

### Compute the Gradients Manually 

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

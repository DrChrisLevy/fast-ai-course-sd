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

```{code-cell} ipython3
import os
os.chdir('/workspace')

# Grab MNIST Data
```

## Data

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
```

```{code-cell} ipython3
print(type(x_train), x_train.shape)
```

We first want to start with basic python objects like lists instead of working directly with Numpy.

+++

## Slicing and Batching

```{code-cell} ipython3
lst1 = list(x_train[0]) # 
vals = lst1[200:210]
vals
```

Here is a nice function to yield batches (it's a generator):

```{code-cell} ipython3
def chunks(x, batch_size):
    for i in range(0, len(x), batch_size):
        yield x[i:i+batch_size]
    
```

```{code-cell} ipython3
chunks(vals, 2)
```

```{code-cell} ipython3
list(chunks(vals, 2))
```

```{code-cell} ipython3
for chunk in chunks(vals, 2):
    print(chunk)
```

```{code-cell} ipython3
for chunk in chunks(vals, 2):
    print(chunk)
    break
```

```{code-cell} ipython3
[c for c in chunks(vals, 2)]
```

Notice that it always starts from the beginning.

Now, what if we wanted to reshape that `lst1`
of 784 pixels into a 28 by 28 image tensor with one channel (with out using Numpy reshape)

```{code-cell} ipython3
len(lst1)
```

```{code-cell} ipython3
plt.imshow(list(chunks(lst1,28)))
```

Now lets look at another cool thing for efficient looping, [itertools](https://docs.python.org/3/library/itertools.html).

Note that iterators and generators are not the same [thing](https://www.codingninjas.com/codestudio/library/iterators-and-generators-in-python)

```{code-cell} ipython3
from itertools import islice
```

```{code-cell} ipython3
vals
```

[iter](https://docs.python.org/3/library/functions.html#iter)

```{code-cell} ipython3
vals_iter = iter(vals)
[x for x in vals_iter]
```

```{code-cell} ipython3
[x for x in vals_iter] # we have already went through it all
```

```{code-cell} ipython3
list(islice(vals,0,7,2)) # not using the iter here. start, stop, step
```

```{code-cell} ipython3
iter_vals = iter(vals)
for i in iter_vals:
    print(i)
```

```{code-cell} ipython3
iter_vals = iter(vals)
for i in iter_vals:
    if i > 0:
        break
```

```{code-cell} ipython3
for i in iter_vals:
    print(i)
```

```{code-cell} ipython3
for i in iter_vals:
    print(i)
```

So `iter` can be exhausted.

```{code-cell} ipython3
it = iter(vals)
batch_size = 5
islice(it, batch_size)
```

```{code-cell} ipython3
list(islice(it, batch_size))
```

```{code-cell} ipython3
list(islice(it, batch_size))
```

```{code-cell} ipython3
list(islice(it, batch_size))
```

```{code-cell} ipython3
it = iter(lst1)
#img = list(iter(lambda: list(islice(it, 28)), []))
```

```{code-cell} ipython3
it = iter(lst1)
img = list(iter(lambda: list(islice(it, 28)), [])) # lol, I'm having a hard time seeing what this one does!
```

```{code-cell} ipython3
plt.imshow(img)
```

## Matrix and Tensor

```{code-cell} ipython3
img[20][15]
```

Create a simple class that can grab the row and column `x[20, 15]`.

[magic methods or dunder methods](https://www.tutorialsteacher.com/python/magic-methods-in-python)

```{code-cell} ipython3
class Matrix:
    def __init__(self, x):
        self.x = x
    
    def __getitem__(self, loc):
        return self.x[loc[0]][loc[1]]
        
```

```{code-cell} ipython3
m = Matrix(img)
m[20,15]
```

```{code-cell} ipython3
import torch
```

```{code-cell} ipython3
x = torch.tensor([[1,2,3],[4,5,6]])
x
```

```{code-cell} ipython3
x.device
```

```{code-cell} ipython3
x.to('cuda')
```

```{code-cell} ipython3
x.cpu()
```

```{code-cell} ipython3
x.device
```

```{code-cell} ipython3
x.cuda()
```

```{code-cell} ipython3
x.shape
```

```{code-cell} ipython3
x.cpu()
```

```{code-cell} ipython3
x.type()
```

```{code-cell} ipython3
x.dtype
```

```{code-cell} ipython3
import numpy as np
torch.tensor(np.array([1.,2.,3.]))
```

```{code-cell} ipython3
x_train,y_train,x_valid,y_valid = map(torch.tensor, (x_train,y_train,x_valid,y_valid))
```

```{code-cell} ipython3
x_train.shape
```

```{code-cell} ipython3
x_train.reshape((1,2,3)) # have to get the right dimensions
```

```{code-cell} ipython3
x_train.reshape((50000,28,28))
```

Can use `-1` to infer a dimension.

```{code-cell} ipython3
imgs = x_train.reshape((-1,28,28)) # same thing as above
```

```{code-cell} ipython3
plt.imshow(imgs[9])
```

```{code-cell} ipython3
imgs[9,27, 27] # 9th digit in list and grabing a single pixel at 27,27
```

## Matrix Multiplication

```{code-cell} ipython3
torch.manual_seed(1)
weights = torch.randn(784,10)
bias = torch.zeros(10)
torch.set_printoptions(precision=2, linewidth=140, sci_mode=False)
```

```{code-cell} ipython3
m1 = x_valid[:5]
m2 = weights
```

```{code-cell} ipython3
m1.shape, m2.shape
```

### naive triple loop matric mult

```{code-cell} ipython3
m1
```

```{code-cell} ipython3
m2
```

```{code-cell} ipython3

```

It's hard for me to even do the triple loop and keep track of the indices 
even though I know how matrix mult works!

```{code-cell} ipython3
def mat_mul(m1,m2):
    ar,ac = m1.shape
    br,bc = m2.shape
    res = torch.zeros((ar,bc))
    for i in range(ar):
        for j in range(bc):
            for k in range(ac):
                res[i,j] += m1[i,k] * m2[k,j]
    return res
mat_mul(m1,m2)
```

```{code-cell} ipython3
m1 @ m2
```

```{code-cell} ipython3
%timeit mat_mul(m1,m2)
```

## Dot Product Speed Up With Numba

[Numba](https://numba.pydata.org/numba-doc/latest/user/5minguide.html)

```{code-cell} ipython3
def dot(a,b):
    assert len(a) == len(b)
    res = 0.0
    for i in range(len(a)):
        res += a[i] * b[i]
    return res
```

```{code-cell} ipython3
dot(m1[0,:],m2[:,0]) # the value at (0,0) in the tensor
```

```{code-cell} ipython3
%timeit dot(m1[0,:],m2[:,0])
```

```{code-cell} ipython3
!pip install numba
```

```{code-cell} ipython3
from numba import njit # I think only works with numpy
```

```{code-cell} ipython3
@njit
def dot(a,b):
    assert len(a) == len(b)
    res = 0.0
    for i in range(len(a)):
        res += a[i] * b[i]
    return res
```

```{code-cell} ipython3
dot(np.array(m1[0,:]),np.array(m2[:,0]))
```

```{code-cell} ipython3
%timeit dot(np.array(m1[0,:]),np.array(m2[:,0]))
```

```{code-cell} ipython3
m1 = np.array(m1)
m2 = np.array(m2)
def mat_mul(m1,m2):
    ar,ac = m1.shape
    br,bc = m2.shape
    res = torch.zeros((ar,bc))
    for i in range(ar):
        for j in range(bc):
            res[i,j] = dot(m1[i,:],m2[:,j])
    return res
mat_mul(m1,m2)
```

```{code-cell} ipython3
%timeit mat_mul(m1,m2)
```

## Elementwise Operations

```{code-cell} ipython3
a = torch.tensor([10., 6, -4])
b = torch.tensor([2., 8, 7])
a,b
```

```{code-cell} ipython3
a+b
```

```{code-cell} ipython3
a*b
```

```{code-cell} ipython3
(a*b).sum() # dot product
```

```{code-cell} ipython3
(a == b).float()
```

```{code-cell} ipython3
(a > b).float().sum()
```

```{code-cell} ipython3
(a < b).float().mean()
```

Frobenius norm:

$$\| A \|_F = \left( \sum_{i,j=1}^n | a_{ij} |^2 \right)^{1/2}$$

```{code-cell} ipython3
m = torch.tensor([[1.,2.,3.],[4.,-5.,-6.],[7.,-8.,9.]])
m
```

```{code-cell} ipython3
m.abs().square().sum().sqrt()
```

```{code-cell} ipython3
(m*m).sum().sqrt()
```

```{code-cell} ipython3
m1 = torch.tensor(m1)
m2 = torch.tensor(m2)
def mat_mul(m1,m2):
    ar,ac = m1.shape
    br,bc = m2.shape
    res = torch.zeros((ar,bc))
    for i in range(ar):
        for j in range(bc):
            res[i,j] = (m1[i,:] * m2[:,j]).sum()
    return res
mat_mul(m1,m2)
```

```{code-cell} ipython3
%timeit mat_mul(m1,m2)
```

```{code-cell} ipython3
def mat_mul(m1,m2):
    ar,ac = m1.shape
    br,bc = m2.shape
    res = torch.zeros((ar,bc))
    for i in range(ar):
        for j in range(bc):
            res[i,j] = torch.dot(m1[i,:] , m2[:,j])
    return res
mat_mul(m1,m2)
```

```{code-cell} ipython3
%timeit mat_mul(m1,m2)
```

## Broadcasting

```{code-cell} ipython3

```

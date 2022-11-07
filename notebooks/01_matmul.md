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

Broadcasting a scalar over a vector or matrix is quite easy to understand:

```{code-cell} ipython3
x
```

```{code-cell} ipython3
2 * x
```

```{code-cell} ipython3
2 + x # pure math would lose their mind over this. Welcome to broadcasting :)
```

```{code-cell} ipython3
c = torch.tensor([1.,2.,3.])
c
```

```{code-cell} ipython3
c + x
```

```{code-cell} ipython3
c.shape,x.shape
```

```{code-cell} ipython3
c * x
```

The smaller one `c` is "broadcast" across the larger one `x`.
Its like we made `c` to have shape (2,3) and then added.
But it does memory optimized. It does not actually copy it.

```{code-cell} ipython3
t = c.expand_as(x)
```

```{code-cell} ipython3
t.shape
```

This is very cool!

```{code-cell} ipython3
t.storage()
```

```{code-cell} ipython3
t.stride(), t.shape
```

```{code-cell} ipython3
c
```

```{code-cell} ipython3
c.unsqueeze(0), c[None,:] # same thing
```

```{code-cell} ipython3
c.shape, c[None,:].shape
```

You can always skip trailing ':'s. And '...' means '*all preceding dimensions*'

```{code-cell} ipython3
c[None] # add unit axis at the start
```

```{code-cell} ipython3
c[...,None] # add unit axis at the end
```

```{code-cell} ipython3
m
```

```{code-cell} ipython3
c.expand_as(m)
```

```{code-cell} ipython3
c.shape, m.shape
```

```{code-cell} ipython3
c+m
```

## Broadcast Rules

```{code-cell} ipython3
c = 10 * c
c
```

```{code-cell} ipython3
c[:, None]
```

```{code-cell} ipython3
c[None, :]
```

Can we multiply these?

```{code-cell} ipython3
c[None,:].shape, c[:,None].shape
```

When operating on two arrays/tensors, Numpy/PyTorch compares their shapes element-wise. It starts with the **trailing dimensions**, and works its way forward. Two dimensions are **compatible** when

- they are equal, or
- one of them is 1, in which case that dimension is broadcasted to make it the same size

Arrays do not need to have the same number of dimensions.

```{code-cell} ipython3
c[:,None].expand_as(m)
```

```{code-cell} ipython3
c[None,:].expand_as(m)
```

```{code-cell} ipython3
c[:,None] * c[None,:] # you see, it first did ^^^ then multiplied element wise. In a optimized fashion
```

Arrays do not need to have the same number of dimensions. For example, if you have a `256*256*3` array of RGB values, and you want to scale each color in the image by a different value, you can multiply the image by a one-dimensional array with 3 values. Lining up the sizes of the trailing axes of these arrays according to the broadcast rules, shows that they are compatible:

    Image  (3d array): 256 x 256 x 3
    Scale  (1d array):             3
    Result (3d array): 256 x 256 x 3

The [numpy documentation](https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html#general-broadcasting-rules) includes several examples of what dimensions can and can not be broadcast together.

+++

Broadcasting provides a convenient way of taking the outer product (or any other outer operation) of two arrays. The following example shows an outer addition operation of two 1-d arrays:

```{code-cell} ipython3
a = np.array([0.0, 10.0, 20.0, 30.0])
b = np.array([1.0, 2.0, 3.0])
```

```{code-cell} ipython3
a.shape
```

```{code-cell} ipython3
b.shape
```

```{code-cell} ipython3
a[:,None].shape
```

```{code-cell} ipython3
a[:,None]+ b
```

## Matmul with broadcasting

for each row of m1 we want to to something with it and every column of m2.
Wouldnt it be nice if we could just have one loop for the rows of m1.

```{code-cell} ipython3
row = m1[0,:]
columns = m2
```

```{code-cell} ipython3
row.shape
```

```{code-cell} ipython3
columns.shape
```

```{code-cell} ipython3
row[:, None].shape
```

```{code-cell} ipython3

```

```{code-cell} ipython3
def mat_mul(m1,m2):
    ar,ac = m1.shape
    br,bc = m2.shape
    res = torch.zeros((ar,bc))
    for i in range(ar):
        res[i] = (m1[i, :, None] * m2).sum(dim=0) # wow its like magic. Probably have to write this out on a piece of paper with much smaller example
    return res
mat_mul(m1,m2)
```

```{code-cell} ipython3
%timeit mat_mul(m1,m2)
```

## Einstein summation

**TODO**: I basically skipped this and need to come back to it.

[Einstein summation](https://ajcr.net/Basic-guide-to-einsum/) ([`einsum`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)) is a compact representation for combining products and sums in a general way. The key rules are:

- Repeating letters between input arrays means that values along those axes will be multiplied together.
- Omitting a letter from the output means that values along that axis will be summed.

```{code-cell} ipython3
m1.shape,m2.shape
```

```{code-cell} ipython3
# c[i,j] += a[i,k] * b[k,j]
# c[i,j] = (a[i,:] * b[:,j]).sum()
mr = torch.einsum('ik,kj->ikj', m1, m2)
mr.shape
```

```{code-cell} ipython3
mr.sum(1)
```

```{code-cell} ipython3
torch.einsum('ik,kj->ij', m1, m2)
```

```{code-cell} ipython3
def matmul(a,b): return torch.einsum('ik,kj->ij', a, b)
```

```{code-cell} ipython3
%timeit -n 5 _=matmul(x_train, weights)
```

## Pytorch Matrix Mult OP

```{code-cell} ipython3
m1 @ m2
```

```{code-cell} ipython3
%timeit -n 10 torch.matmul(m1,m2)
```

## CUDA

**TODO**: go back and study this part again b/c I did not spend a lot of time on it.

```{code-cell} ipython3
def matmul(grid, a,b,c):
    i,j = grid
    if i < c.shape[0] and j < c.shape[1]:
        tmp = 0.
        for k in range(a.shape[1]):
            tmp += a[i, k] * b[k, j]
        c[i,j] = tmp
```

```{code-cell} ipython3
res = torch.zeros(ar, bc)
matmul((0,0), m1, m2, res)
res
```

```{code-cell} ipython3
def launch_kernel(kernel, grid_x, grid_y, *args, **kwargs):
    for i in range(grid_x):
        for j in range(grid_y): kernel((i,j), *args, **kwargs)
```

```{code-cell} ipython3
res = torch.zeros(ar, bc)
launch_kernel(matmul, ar, bc, m1, m2, res)
res
```

```{code-cell} ipython3
from numba import cuda
```

```{code-cell} ipython3
def matmul(grid, a,b,c):
    i,j = grid
    if i < c.shape[0] and j < c.shape[1]:
        tmp = 0.
        for k in range(a.shape[1]): tmp += a[i, k] * b[k, j]
        c[i,j] = tmp
```

```{code-cell} ipython3
@cuda.jit
def matmul(a,b,c):
    i, j = cuda.grid(2)
    if i < c.shape[0] and j < c.shape[1]:
        tmp = 0.
        for k in range(a.shape[1]): tmp += a[i, k] * b[k, j]
        c[i,j] = tmp
```

go back and see the NB. Never finished this...
https://github.com/fastai/course22p2/blob/master/nbs/01_matmul.ipynb

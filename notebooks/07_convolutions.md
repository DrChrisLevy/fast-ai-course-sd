---
jupytext:
  formats: ipynb,md:myst
  split_at_heading: true
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
#|default_exp conv
```

```{code-cell} ipython3
import os
os.chdir('/workspace')
```

# Convolutions
- to get the miniai library below to behave properly
need to `pip install -e .` within the `/workspace/course22p2` dir

```{code-cell} ipython3
import os
os.chdir('course22p2/')
os.system('pip install -e .')
os.chdir('/workspace')
```

```{code-cell} ipython3
#|export
import pickle,gzip,math,os,time,shutil,torch,matplotlib as mpl, numpy as np
import pandas as pd,matplotlib.pyplot as plt
from pathlib import Path
from torch import tensor
from torch import nn

from torch.utils.data import DataLoader,default_collate
from typing import Mapping

from miniai.training import *
from miniai.datasets import *
```

```{code-cell} ipython3
mpl.rcParams['image.cmap'] = 'gray'
```

```{code-cell} ipython3
path_data = Path('data')
path_gz = path_data/'mnist.pkl.gz'
with gzip.open(path_gz, 'rb') as f: ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
x_train, y_train, x_valid, y_valid = map(tensor, [x_train, y_train, x_valid, y_valid])
```

In the context of an image, a feature is a visually distinctive attribute. For example, the number 7 is characterized by a horizontal edge near the top of the digit, and a top-right to bottom-left diagonal edge underneath that.

It turns out that finding the edges in an image is a very common task in computer vision, and is surprisingly straightforward. To do it, we use a *convolution*. A convolution requires nothing more than multiplication, and addition.

+++

### Understanding the Convolution Equations

+++

To explain the math behind convolutions, fast.ai student Matt Kleinsmith came up with the very clever idea of showing [CNNs from different viewpoints](https://medium.com/impactai/cnns-from-different-viewpoints-fab7f52d159c).

Here's the input:

+++

<img alt="The image" width="75" src="course22p2/nbs/images/att_00032.png">

+++

Here's our kernel:

+++

<img alt="The kernel" width="55" src="images/att_00033.png">

+++

Since the filter fits in the image four times, we have four results:

+++

<img alt="The activations" width="52" src="images/att_00034.png">

+++

<img alt="Applying the kernel" width="366" caption="Applying the kernel" id="apply_kernel" src="images/att_00035.png">

+++

<img alt="The equation" width="436" caption="The equation" id="eq_view" src="images/att_00036.png">

```{code-cell} ipython3
x_imgs = x_train.view(-1,28,28)
xv_imgs = x_valid.view(-1,28,28)
```

```{code-cell} ipython3
mpl.rcParams['figure.dpi'] = 30
```

```{code-cell} ipython3
im3 = x_imgs[7]
show_image(im3);
```

```{code-cell} ipython3
# convolutional kernel
top_edge = tensor([[-1,-1,-1],
                   [ 0, 0, 0],
                   [ 1, 1, 1]]).float()
```

We're going to call this our kernel (because that's what fancy computer vision researchers call these).

```{code-cell} ipython3
show_image(top_edge, noframe=False);
```

The filter will take any window of size 3×3 in our images, and if we name the pixel values like this:

$$\begin{matrix} a1 & a2 & a3 \\ a4 & a5 & a6 \\ a7 & a8 & a9 \end{matrix}$$

it will return $-a1-a2-a3+a7+a8+a9$.

```{code-cell} ipython3
df = pd.DataFrame(im3[:130,:230])
df.style.format(precision=2).set_properties(**{'font-size':'7pt'}).background_gradient('Greys')
```

```{code-cell} ipython3
df = pd.DataFrame(im3[:13,:23])
df.style.format(precision=2).set_properties(**{'font-size':'7pt'}).background_gradient('Greys')
```

```{code-cell} ipython3
(im3[3:6,14:17] * top_edge).sum()
```

```{code-cell} ipython3
(im3[7:10,14:17] * top_edge).sum()
```

```{code-cell} ipython3
def apply_kernel(row, col, kernel): # the row,col is the center of where the 3by3 kernel is placed
    return (im3[row-1:row+2,col-1:col+2] * kernel).sum()
```

```{code-cell} ipython3
apply_kernel(4,15,top_edge)
```

<img src="images/chapter9_nopadconv.svg" id="nopad_conv" caption="Applying a kernel across a grid" alt="Applying a kernel across a grid" width="400">

```{code-cell} ipython3
[[(i,j) for j in range(5)] for i in range(5)]
```

```{code-cell} ipython3
show_image(im3)
```

```{code-cell} ipython3
im3.shape
```

```{code-cell} ipython3
apply_kernel(26,26,top_edge)
```

```{code-cell} ipython3
rng = range(1,27)
top_edge3 = tensor([[apply_kernel(i,j,top_edge) for j in rng] for i in rng])
top_edge3.shape
```

after applying the kernel (3X3) to an image (28X28)
we end up with an image of shape (28-3+1,28-3+1)

```{code-cell} ipython3
top_edge3.shape
```

```{code-cell} ipython3
show_image(top_edge3)
```

```{code-cell} ipython3
left_edge = tensor([[-1,0,1],
                    [-1,0,1],
                    [-1,0,1]]).float()
```

```{code-cell} ipython3
show_image(left_edge, noframe=False);
```

```{code-cell} ipython3
left_edge3 = tensor([[apply_kernel(i,j,left_edge) for j in rng] for i in rng])
show_image(left_edge3);
```

<img alt="Result of applying a 3×3 kernel to a 4×4 image" width="782" caption="Result of applying a 3×3 kernel to a 4×4 image (courtesy of Vincent Dumoulin and Francesco Visin)" id="three_ex_four_conv" src="images/att_00028.png">

+++

### Convolutions in PyTorch

```{code-cell} ipython3
import torch.nn.functional as F
import torch
```

What to do if you have [2 months to complete your thesis](https://github.com/Yangqing/caffe/wiki/Convolution-in-Caffe:-a-memo)? Use [im2col](https://hal.inria.fr/inria-00112631/).

+++

![image.png](attachment:image.png)

+++

Here's a sample [numpy implementation](https://github.com/3outeille/CNNumpy/blob/5394f13e7ed67a808a3e39fd381f168825d65ff5/src/fast/utils.py#L360).

```{code-cell} ipython3
im3.shape
```

```{code-cell} ipython3
inp = im3[None,None, :, :] # batch dim, channel depth dim, h, w
inp.shape
```

Here we doing a convolution using the unrolled input features method.
Like discussed above. This cool trick using matrix mult and `F.unfold`.

```{code-cell} ipython3
F.unfold(inp, (3,3)).shape
```

```{code-cell} ipython3
inp_unf = F.unfold(inp, (3,3))[0]
inp_unf.shape
```

```{code-cell} ipython3
left_edge.shape
```

```{code-cell} ipython3
w = left_edge.view(-1)
w.shape
```

```{code-cell} ipython3
out_unf = w@inp_unf
out_unf.shape
```

```{code-cell} ipython3
out = out_unf.view(26,26)
show_image(out);
```

```{code-cell} ipython3
%timeit -n 1 tensor([[apply_kernel(i,j,left_edge) for j in rng] for i in rng]);
```

```{code-cell} ipython3
%timeit -n 100 (w@F.unfold(inp, (3,3))[0]).view(26,26);
```

```{code-cell} ipython3
left_edge[None,None].shape
```

```{code-cell} ipython3
%timeit -n 100 F.conv2d(inp, left_edge[None,None])
```

```{code-cell} ipython3

```

```{code-cell} ipython3
diag1_edge = tensor([[ 0,-1, 1],
                     [-1, 1, 0],
                     [ 1, 0, 0]]).float()
```

```{code-cell} ipython3
show_image(diag1_edge, noframe=False);
```

```{code-cell} ipython3
diag2_edge = tensor([[ 1,-1, 0],
                     [ 0, 1,-1],
                     [ 0, 0, 1]]).float()
```

```{code-cell} ipython3
show_image(diag2_edge, noframe=False)
```

```{code-cell} ipython3
xb = x_imgs[:16][:,None]
xb.shape
```

```{code-cell} ipython3
edge_kernels = torch.stack([left_edge, top_edge, diag1_edge, diag2_edge])[:,None]
edge_kernels.shape
```

```{code-cell} ipython3
batch_features = F.conv2d(xb, edge_kernels)
batch_features.shape
```

The output shape shows we gave 64 images in the mini-batch, 4 kernels, and 26×26 edge maps (we started with 28×28 images, but lost one pixel from each side as discussed earlier). We can see we get the same results as when we did this manually:

```{code-cell} ipython3
img0 = xb[1,0]
show_image(img0);
```

```{code-cell} ipython3
show_images([batch_features[1,i] for i in range(4)])
```

Quick Practice:

```{code-cell} ipython3
#batch of 32 images
x = x_train[30:62].view(32,28,28) # x = x_train[30:62].view(-1,28,28)
x.shape
```

We need to add the channel dimension b/c thats what F.conv2d expects

```{code-cell} ipython3
x = x[:, None, :,  :]
```

```{code-cell} ipython3
x.shape
```

Okay so we have our batch of 32 images and now we want to apply some 
convolutional kernel over it.

```{code-cell} ipython3
# conv kernels
filters = tensor([[[[-1.,  0.,  1.],
          [-1.,  0.,  1.],
          [-1.,  0.,  1.]]],


        [[[-1., -1., -1.],
          [ 0.,  0.,  0.],
          [ 1.,  1.,  1.]]],


        [[[ 0., -1.,  1.],
          [-1.,  1.,  0.],
          [ 1.,  0.,  0.]]],


        [[[ 1., -1.,  0.],
          [ 0.,  1., -1.],
          [ 0.,  0.,  1.]]]])
filters.shape # out_channels, in_channels/groups, kernel_height, kernal_width
```

```{code-cell} ipython3
F.conv2d(x, filters).shape
```

Also see [nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)

```{code-cell} ipython3
m = nn.Conv2d(3, 32, 5, stride=2) # n_channels, out_channels, kernel_size, stride=1
```

```{code-cell} ipython3
[x for x in m.parameters()][0].shape  # Kernel
```

```{code-cell} ipython3
[x for x in m.parameters()][1].shape # bias term
```

```{code-cell} ipython3
X = torch.randn(64, 3, 512, 512) # batch of 64 rgb images
```

```{code-cell} ipython3
m(X).shape
```

### Strides and Padding

+++

With appropriate padding, we can ensure that the output activation map is the same size as the original image.

+++

<img src="images/chapter9_padconv.svg" id="pad_conv" caption="A convolution with padding" alt="A convolution with padding" width="600">

+++

With a 5×5 input, 4×4 kernel, and 2 pixels of padding, we end up with a 6×6 activation map.

+++

<img alt="A 4×4 kernel with 5×5 input and 2 pixels of padding" width="783" caption="A 4×4 kernel with 5×5 input and 2 pixels of padding (courtesy of Vincent Dumoulin and Francesco Visin)" id="four_by_five_conv" src="images/att_00029.png">

+++

If we add a kernel of size `ks` by `ks` (with `ks` an odd number), the necessary padding on each side to keep the same shape is `ks//2`.

We could move over two pixels after each kernel application. This is known as a *stride-2* convolution.

+++

<img alt="A 3×3 kernel with 5×5 input, stride-2 convolution, and 1 pixel of padding" width="774" caption="A 3×3 kernel with 5×5 input, stride-2 convolution, and 1 pixel of padding (courtesy of Vincent Dumoulin and Francesco Visin)" id="three_by_five_conv" src="images/att_00030.png">

+++

## Creating the CNN

```{code-cell} ipython3
n,m = x_train.shape
c = y_train.max()+1
nh = 50
```

```{code-cell} ipython3
n,m,c
```

```{code-cell} ipython3
# reminder what we had used before
model = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,10))
```

```{code-cell} ipython3
broken_cnn = nn.Sequential(
    nn.Conv2d(1,30, kernel_size=3, padding=1), #in_channels, out_channels, kernel_size, stride=1, padding
    nn.ReLU(),
    nn.Conv2d(30,10, kernel_size=3, padding=1) #in_channels, out_channels, kernel_size, stride=1, padding
)
```

```{code-cell} ipython3
xb.shape
```

```{code-cell} ipython3
broken_cnn(xb).shape # not right shape to predict 10 digits classifier 
```

```{code-cell} ipython3
#|export
def conv(ni, nf, ks=3, stride=2, act=True):
    res = nn.Conv2d(ni, nf, stride=stride, kernel_size=ks, padding=ks//2) #in_channels, out_channels, kernel_size, stride=1, padding
    if act:
        res = nn.Sequential(res, nn.ReLU())
    return res
```

```{code-cell} ipython3

```

Refactoring parts of your neural networks like this makes it much less likely you'll get errors due to inconsistencies in your architectures, and makes it more obvious to the reader which parts of your layers are actually changing.

```{code-cell} ipython3
simple_cnn = nn.Sequential(
    conv(1 ,4),            #14x14
    conv(4 ,8),            #7x7
    conv(8 ,16),           #4x4
    conv(16,16),           #2x2
    conv(16,10, act=False), #1x1
    nn.Flatten(),
)
```

```{code-cell} ipython3
simple_cnn(xb).shape
```

```{code-cell} ipython3
x_imgs = x_train.view(-1,1,28,28)
xv_imgs = x_valid.view(-1,1,28,28)
train_ds,valid_ds = Dataset(x_imgs, y_train),Dataset(xv_imgs, y_valid)
```

```{code-cell} ipython3

```

```{code-cell} ipython3
for x in train_ds:
    break
```

```{code-cell} ipython3
x[0].shape, x[1]
```

```{code-cell} ipython3
#|export
def_device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

def to_device(x, device=def_device):
    if isinstance(x, Mapping): return {k:v.to(device) for k,v in x.items()}
    return type(x)(o.to(device) for o in x)

def collate_device(b): return to_device(default_collate(b))
```

```{code-cell} ipython3
from torch import optim

bs = 256
lr = 0.4
train_dl,valid_dl = get_dls(train_ds, valid_ds, bs, collate_fn=collate_device)
opt = optim.SGD(simple_cnn.parameters(), lr=lr)
```

```{code-cell} ipython3
loss,acc = fit(5, simple_cnn.to(def_device), F.cross_entropy, opt, train_dl, valid_dl)
```

```{code-cell} ipython3
opt = optim.SGD(simple_cnn.parameters(), lr=lr/4)
loss,acc = fit(5, simple_cnn.to(def_device), F.cross_entropy, opt, train_dl, valid_dl)
```

```{code-cell} ipython3
??fit
```

### Understanding Convolution Arithmetic

+++

In an input of size `64x1x28x28` the axes are `batch,channel,height,width`. This is often represented as `NCHW` (where `N` refers to batch size). Tensorflow, on the other hand, uses `NHWC` axis order (aka "channels-last"). Channels-last is faster for many models, so recently it's become more common to see this as an option in PyTorch too.

We have 1 input channel, 4 output channels, and a 3×3 kernel.

```{code-cell} ipython3
conv1 = simple_cnn[0][0]
conv1.weight.shape
```

```{code-cell} ipython3
conv1.bias.shape
```

The *receptive field* is the area of an image that is involved in the calculation of a layer. *conv-example.xlsx* shows the calculation of two stride-2 convolutional layers using an MNIST digit. Here's what we see if we click on one of the cells in the *conv2* section, which shows the output of the second convolutional layer, and click *trace precedents*.

+++

<img alt="Immediate precedents of conv2 layer" width="308" caption="Immediate precedents of Conv2 layer" id="preced1" src="images/att_00068.png">

+++

The blue highlighted cells are its *precedents*—that is, the cells used to calculate its value. These cells are the corresponding 3×3 area of cells from the input layer (on the left), and the cells from the filter (on the right). Click *trace precedents* again:

+++

<img alt="Secondary precedents of conv2 layer" width="601" caption="Secondary precedents of Conv2 layer" id="preced2" src="images/att_00069.png">

+++

In this example, we have just two convolutional layers. We can see that a 7×7 area of cells in the input layer is used to calculate the single green cell in the Conv2 layer. This is the *receptive field*

The deeper we are in the network (specifically, the more stride-2 convs we have before a layer), the larger the receptive field for an activation in that layer.

+++

## Color Images

+++

A colour picture is a rank-3 tensor:

```{code-cell} ipython3
from torchvision.io import read_image
```

```{code-cell} ipython3
im = read_image('images/grizzly.jpg')
im.shape
```

```{code-cell} ipython3
show_image(im.permute(1,2,0));
```

```{code-cell} ipython3
_,axs = plt.subplots(1,3)
for bear,ax,color in zip(im,axs,('Reds','Greens','Blues')): show_image(255-bear, ax=ax, cmap=color)
```

<img src="images/chapter9_rgbconv.svg" id="rgbconv" caption="Convolution over an RGB image" alt="Convolution over an RGB image" width="550">

+++

These are then all added together, to produce a single number, for each grid location, for each output feature.

+++

<img src="images/chapter9_rgb_conv_stack.svg" id="rgbconv2" caption="Adding the RGB filters" alt="Adding the RGB filters" width="500">

+++

We have `ch_out` filters like this, so in the end, the result of our convolutional layer will be a batch of images with `ch_out` channels.

+++

## Export -

```{code-cell} ipython3
import nbdev; nbdev.nbdev_export()
```

```{code-cell} ipython3

```

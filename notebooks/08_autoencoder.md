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

# Autoencoders

```{code-cell} ipython3
import torch.cuda
import torch
#torch.tensor(1).cuda()
```

```{code-cell} ipython3
import pickle,gzip,math,os,time,shutil,torch,matplotlib as mpl,numpy as np,matplotlib.pyplot as plt
import fastcore.all as fc
from collections.abc import Mapping
from pathlib import Path
from operator import attrgetter,itemgetter
from functools import partial

from torch import tensor,nn,optim
from torch.utils.data import DataLoader,default_collate
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from datasets import load_dataset,load_dataset_builder

from fastprogress import progress_bar,master_bar
from miniai.datasets import *
from miniai.training import *
from miniai.conv import *
```

```{code-cell} ipython3
from fastcore.test import test_close

torch.set_printoptions(precision=2, linewidth=140, sci_mode=False)
torch.manual_seed(1)
mpl.rcParams['image.cmap'] = 'gray'

import logging
logging.disable(logging.WARNING)
```

## Data

```{code-cell} ipython3
x,y = 'image','label'
name = "fashion_mnist"
dsd = load_dataset(name, ignore_verifications=True)
```

```{code-cell} ipython3
@inplace
def transformi(b): b[x] = [TF.to_tensor(o) for o in b[x]]
```

```{code-cell} ipython3
bs = 256
tds = dsd.with_transform(transformi)
```

```{code-cell} ipython3
ds = tds['train']
img = ds[0]['image']
show_image(img, figsize=(1,1));
```

```{code-cell} ipython3
cf = collate_dict(ds)
```

```{code-cell} ipython3
def collate_(b): return to_device(cf(b))
def data_loaders(dsd, bs, **kwargs): return {k:DataLoader(v, bs, **kwargs) for k,v in dsd.items()}
```

```{code-cell} ipython3
dls = data_loaders(tds, bs, num_workers=4, collate_fn=collate_)
```

```{code-cell} ipython3
dt = dls['train']
dv = dls['test']

xb,yb = next(iter(dt))
```

```{code-cell} ipython3
xb.shape
```

```{code-cell} ipython3
yb.shape
```

```{code-cell} ipython3
labels = ds.features[y].names
lbl_getter = itemgetter(*yb[:16])
titles = lbl_getter(labels)
```

```{code-cell} ipython3
mpl.rcParams['figure.dpi'] = 70
show_images(xb[:16], imsize=1.7, titles=titles)
```

## Warmup - classify

```{code-cell} ipython3
from torch import optim

bs = 256
lr = 0.4
```

```{code-cell} ipython3
cnn = nn.Sequential(
    conv(1 ,4),            #14x14
    conv(4 ,8),            #7x7
    conv(8 ,16),           #4x4
    conv(16,16),           #2x2
    conv(16,10, act=False),
    nn.Flatten()).to(def_device)
```

```{code-cell} ipython3
opt = optim.SGD(cnn.parameters(), lr=lr)
loss,acc = fit(5, cnn, F.cross_entropy, opt, dt, dv)
```

## Autoencoder

```{code-cell} ipython3
def deconv(ni, nf, ks=3, act=True):
    layers = [nn.UpsamplingNearest2d(scale_factor=2),
              nn.Conv2d(ni, nf, stride=1, kernel_size=ks, padding=ks//2)]
    if act: layers.append(nn.ReLU())
    return nn.Sequential(*layers)
```

```{code-cell} ipython3
def eval(model, loss_func, valid_dl, epoch=0):
    model.eval()
    with torch.no_grad():
        tot_loss,count = 0.,0
        for xb,_ in valid_dl:
            pred = model(xb)
            n = len(xb)
            count += n
            tot_loss += loss_func(pred,xb).item()*n
    print(epoch, f'{tot_loss/count:.3f}')
```

```{code-cell} ipython3
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb,_ in train_dl:
            loss = loss_func(model(xb), xb)
            loss.backward()
            opt.step()
            opt.zero_grad()
        eval(model, loss_func, valid_dl, epoch)
```

```{code-cell} ipython3
ae = nn.Sequential(   #28x28
    nn.ZeroPad2d(2),  #32x32
    conv(1,2),        #16x16
    conv(2,4),        #8x8
#     conv(4,8),        #4x4
#     deconv(8,4),      #8x8
    deconv(4,2),      #16x16
    deconv(2,1, act=False), #32x32
    nn.ZeroPad2d(-2), #28x28
    nn.Sigmoid()
).to(def_device)
```

```{code-cell} ipython3
eval(ae, F.mse_loss, dv)
```

```{code-cell} ipython3
opt = optim.AdamW(ae.parameters(), lr=0.01)
fit(5, ae, F.mse_loss, opt, dt, dv)
```

```{code-cell} ipython3
opt = optim.AdamW(ae.parameters(), lr=0.01)
fit(5, ae, F.l1_loss, opt, dt, dv)
```

```{code-cell} ipython3
p = ae(xb)
show_images(p[:16].data.cpu(), imsize=1.5)
```

```{code-cell} ipython3
p = ae(xb)
show_images(p[:16].data.cpu(), imsize=1.5)
```

```{code-cell} ipython3
show_images(xb[:16].data.cpu(), imsize=1.5)
```

```{code-cell} ipython3

```

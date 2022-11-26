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
from datasets import load_dataset

import torch
from torchvision import transforms
```

```{code-cell} ipython3
dsd = load_dataset("beans")
```

```{code-cell} ipython3
dsd
```

```{code-cell} ipython3
dsd['train']
```

```{code-cell} ipython3
dsd['train']['image'][0]
```

```{code-cell} ipython3
train, val = dsd['train'], dsd['validation']
```

```{code-cell} ipython3
train.features
```

```{code-cell} ipython3
train.features
```

```{code-cell} ipython3
for x in train:
    break
```

```{code-cell} ipython3
x
```

Lets create a simple torch dataset for practice from this HF dataset.

```{code-cell} ipython3
from torch.utils.data import Dataset
```

```{code-cell} ipython3
class MyDS(Dataset):
    def __init__(self, hf_ds):
        self.hf_ds = hf_ds
    
    def __len__(self):
        return len(self.hf_ds)
    
    def __getitem__(self, i):
        return self.hf_ds[i]
```

```{code-cell} ipython3
train[10]
```

```{code-cell} ipython3
MyDS(train)[10]
```

You see, we don't even need to create a torch dataset here because the HF dataset already basically behaves like that.

Lets create a torch DataLoader from the HF dataset.

```{code-cell} ipython3
from torch.utils.data import DataLoader
```

```{code-cell} ipython3
next(iter(train)) # train is the HF dataset
```

```{code-cell} ipython3
def collate_fn(batch):
    return (torch.stack([transforms.ToTensor()(b['image']) for b in batch]),
            torch.tensor([b['labels'] for b in batch]))
```

```{code-cell} ipython3
train_dl = DataLoader(train, batch_size=32, shuffle=True, sampler=None,
           batch_sampler=None, collate_fn=collate_fn)
```

```{code-cell} ipython3
for x,y in train_dl:
    break
x.shape,y.shape
```

We can also achieve the same thing within HF dataset framework instead of using a collate_fn.

```{code-cell} ipython3
def transfm(batch):
    batch['image'] = [transforms.ToTensor()(o) for o in batch['image']]
    return batch
```

```{code-cell} ipython3
next(iter(train.with_transform(transfm)))
```

```{code-cell} ipython3
dl = DataLoader(train.with_transform(transfm), batch_size=16)
```

```{code-cell} ipython3
next(iter(dl))
```

```{code-cell} ipython3
next(iter(dl))['image'].shape
```

Now with inplace with decorators

```{code-cell} ipython3
def _transfm(batch):
    batch['image'] = [transforms.ToTensor()(o) for o in batch['image']]
    # NOTE: It does not return anything!
```

```{code-cell} ipython3
def inplace(f):
    def _f(b):
        f(b)
        return b
    return _f
```

```{code-cell} ipython3
transformi = inplace(_transfm) # returns a function
```

```{code-cell} ipython3
train.with_transform(transformi)[0]
```

Same thing can be done this decorator

```{code-cell} ipython3
@inplace
def transformi(batch):
    batch['image'] = [transforms.ToTensor()(o) for o in batch['image']]
```

```{code-cell} ipython3
tdsf = train.with_transform(transformi)
r = tdsf[0]
r
```

Note that HF returns a lot of dict objects. Many other
platforms work with tuples instead. Here is a 
nifty little function to convert to the tuple convention.

```{code-cell} ipython3
from operator import itemgetter
```

```{code-cell} ipython3
d = dict(a=1,b=2,c=3)
ig = itemgetter('a','c')
ig(d)
```

```{code-cell} ipython3
ig = itemgetter('image', 'labels')
```

```{code-cell} ipython3
ig(r)
```

Does not have to just work with dicts.
Works with any objects with `__getitem__`

```{code-cell} ipython3
class D:
    def __getitem__(self, k):
        return 1 if k=='a' else 2 if k=='b' else 3
```

```{code-cell} ipython3
d = D()
d[1], d[None], d['a'], d['b']
```

```{code-cell} ipython3
ig = itemgetter('a', 'b', 'c')
```

```{code-cell} ipython3
list(tdsf.features)
```

```{code-cell} ipython3
from torch.utils.data import default_collate

batch = dict(a=[1],b=[2]), dict(a=[3],b=[4])
default_collate(batch)
```

```{code-cell} ipython3
def collate_dict(ds):
    get = itemgetter(*ds.features)
    def _f(b): return get(default_collate(b))
    return _f
```

```{code-cell} ipython3
dlf = DataLoader(tdsf, batch_size=4, collate_fn=collate_dict(tdsf))
next(iter(dlf)) # now returning tuple
```

```{code-cell} ipython3
tdsf
```

## Plotting images
[see the course notebook](https://github.com/fastai/course22p2/blob/master/nbs/05_datasets.ipynb)

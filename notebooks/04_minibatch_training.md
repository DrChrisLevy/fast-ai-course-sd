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
import pickle,gzip,math,os,time,shutil,torch,matplotlib as mpl,numpy as np,matplotlib.pyplot as plt
from pathlib import Path
from torch import tensor,nn
import torch.nn.functional as F
from fastcore.test import test_close
import os
os.chdir('/workspace')

torch.set_printoptions(precision=2, linewidth=140, sci_mode=False)
torch.manual_seed(1)
mpl.rcParams['image.cmap'] = 'gray'

MNIST_URL='https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/data/mnist.pkl.gz?raw=true'
path_data = Path('data')
path_data.mkdir(exist_ok=True)
path_gz = path_data/'mnist.pkl.gz'

from urllib.request import urlretrieve
if not path_gz.exists():
    urlretrieve(MNIST_URL, path_gz)
with gzip.open(path_gz, 'rb') as f: ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
x_train, y_train, x_valid, y_valid = map(tensor, [x_train, y_train, x_valid, y_valid])
```

Some math to go along with this notebook

![](../imgs/log_softmax_1.jpg)
![](../imgs/log_softmax_2.jpg)

```{code-cell} ipython3
x_train = x_train[:512]
y_train = y_train[:512]
```

```{code-cell} ipython3
y = y_train
```

```{code-cell} ipython3
n,m = x_train.shape
c = y_train.max()+1
nh = 50
```

```{code-cell} ipython3
class Model(nn.Module):
    def __init__(self, n_in, nh, n_out):
        super().__init__()
        self.layers = [nn.Linear(n_in,nh), nn.ReLU(), nn.Linear(nh,n_out)]
        
    def __call__(self, x):
        for l in self.layers: x = l(x)
        return x
```

```{code-cell} ipython3
model = Model(m, nh, 10) # now the output of the last layer is dimension 10
pred = model(x_train)
pred.shape
```

Everything above was from last time (see notebook 03 back prop).

```{code-cell} ipython3
pred[:10]
```

Lets look at the first prediction and take the `softmax` of it

```{code-cell} ipython3
def softmax(x):
    return torch.exp(x) / torch.exp(x).sum()
```

```{code-cell} ipython3
softmax(pred[0])
```

```{code-cell} ipython3
softmax(pred[0]).sum() #should sum to 1
```

But that above definition will only work for a vector `pred`. What if we wanted to pass the batch `pred`.

```{code-cell} ipython3
torch.exp(pred).shape
```

```{code-cell} ipython3
torch.exp(pred).sum(dim=1).shape
```

We need to make the sizes proper for broadcasting:

```{code-cell} ipython3
torch.exp(pred).sum(dim=1)[:, None].shape
```

```{code-cell} ipython3
torch.exp(pred) / torch.exp(pred).sum(dim=1)[:, None]
```

```{code-cell} ipython3
(torch.exp(pred) / torch.exp(pred).sum(dim=1)[:, None]).sum(dim=1)
```

```{code-cell} ipython3
def softmax(x):
    return torch.exp(x) / torch.exp(x).sum(dim=1)[:, None]
```

```{code-cell} ipython3
softmax(pred)
```

In practice we need the `log` of the [softmax](https://ogunlao.github.io/2020/04/26/you_dont_really_know_softmax.html) because it has improvements (numerical stability etc)

```{code-cell} ipython3
def log_softmax(x):
    return torch.log(torch.exp(x) / torch.exp(x).sum(dim=1)[:, None])
```

```{code-cell} ipython3
log_softmax(pred)
```

Using `log` rules we can rewrite this

```{code-cell} ipython3
def log_softmax(x):
    return x - torch.log(torch.exp(x).sum(dim=1)[:, None])
```

```{code-cell} ipython3
log_softmax(pred)
```

Then, there is a way to compute the log of the sum of exponentials in a more stable way, called the [LogSumExp trick](https://en.wikipedia.org/wiki/LogSumExp). The idea is to use the following formula:

$$\log \left ( \sum_{j=1}^{n} e^{x_{j}} \right ) = \log \left ( e^{a} \sum_{j=1}^{n} e^{x_{j}-a} \right ) = a + \log \left ( \sum_{j=1}^{n} e^{x_{j}-a} \right )$$

where a is the maximum of the $x_{j}$.

```{code-cell} ipython3
def logsumexp(x): # want to work for batch of vectors x
    a = x.max(dim=1)[0][:, None] # max entry for each row of shape [batch_size, 1]
    return a + torch.exp(x-a).sum(dim=1).log()[:,None]
```

```{code-cell} ipython3
logsumexp(pred).shape
```

```{code-cell} ipython3
def log_softmax(x):
     return x - logsumexp(x)
```

```{code-cell} ipython3
log_softmax(pred)
```

```{code-cell} ipython3
sm_pred = log_softmax(pred)
sm_pred
```

The cross entropy loss for some target $x$ and some prediction $p(x)$ is given by:

$$ -\sum x\, \log p(x) $$

But since our $x$s are 1-hot encoded, this can be rewritten as $-\log(p_{i})$ where i is the index of the desired target.

This can be done using numpy-style [integer array indexing](https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#integer-array-indexing). Note that PyTorch supports all the tricks in the advanced indexing methods discussed in that link.

```{code-cell} ipython3
sm_pred[:5]
```

```{code-cell} ipython3
y[:5]
```

```{code-cell} ipython3
def nll(pred, y):
    return -pred[range(len(pred)), y].mean()
```

```{code-cell} ipython3
test_close(F.log_softmax(pred, dim=1), sm_pred)
```

```{code-cell} ipython3
nll(sm_pred, y)
```

```{code-cell} ipython3
F.cross_entropy(pred, y)
```

```{code-cell} ipython3
F.nll_loss(F.log_softmax(pred, dim=1), y) # same as F.cross_entropy
```

So in Pytorch F.cross_entropy applies the log_softmax and then the nll_loss.
This is different from Tensorflow. See [here](https://stackoverflow.com/questions/72622202/why-is-the-tensorflow-and-pytorch-crossentropy-loss-returns-different-values-for).

+++

# Basic training loop

```{code-cell} ipython3
with gzip.open(path_gz, 'rb') as f: ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
x_train, y_train, x_valid, y_valid = map(tensor, [x_train, y_train, x_valid, y_valid])
```

```{code-cell} ipython3
class Model(nn.Module):
    def __init__(self, n_in, nh, n_out):
        super().__init__()
        self.layers = [nn.Linear(n_in,nh), nn.ReLU(), nn.Linear(nh,n_out)]
        
    def forward(self, x):
        for l in self.layers: x = l(x)
        return x
```

```{code-cell} ipython3
model = Model(784, 50, 10)
```

```{code-cell} ipython3
def accuracy(ypred, ytrue): return (torch.argmax(ypred, dim=1)==ytrue).float().mean()
```

```{code-cell} ipython3
bs = 512
lr = 0.5
for epoch in range(3):
    for i in range(0, len(x_train), bs):
        ypred = model(x_train[i:i+bs]) # have not been put through softmax etc.
        ytrue = y_train[i:i+bs]
        loss = F.cross_entropy(ypred, ytrue)
        loss.backward()
        with torch.no_grad():
            for l in model.layers:
                if hasattr(l, 'weight'):
                    l.weight -= lr*l.weight.grad
                    l.bias -= lr*l.bias.grad
                    l.weight.grad.zero_()
                    l.bias.grad.zero_()
        print(accuracy(ypred, ytrue), loss)
```

## Using parameters

+++

It would be nice if we could access all the model parameters.
Read more about modules [here](https://pytorch.org/docs/stable/notes/modules.html#a-simple-custom-module)

```{code-cell} ipython3
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(5,5)
        self.layer2 = nn.Linear(5,10)
    
    def forward(x):
        return self.layer2(self.layer1(x))
```

```{code-cell} ipython3
dm = DummyModel()
```

```{code-cell} ipython3
list(dm.named_modules())
```

This module `DummyModel` is composed of two “children” or “submodules” (`layer1` and `layer1`) that define the layers of the neural network and are utilized for computation within the module’s `forward()` method. Immediate children of a module can be iterated through via a call to `children()` or `named_children()`:

```{code-cell} ipython3
list(dm.named_children())
```

For any given module, its parameters consist of its direct parameters as well as the parameters of all submodules. This means that calls to `parameters()` and `named_parameters()` will recursively include child parameters, allowing for convenient optimization of all parameters within the network:

```{code-cell} ipython3
list(dm.parameters())
```

```{code-cell} ipython3
class MLP(nn.Module):
    def __init__(self, n_in, nh, n_out):
        super().__init__()
        self.layer1 = nn.Linear(n_in, nh)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(nh, n_out)
        
    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))
```

```{code-cell} ipython3
model = MLP(784, 50, 10)
```

```{code-cell} ipython3
def fit():
    bs = 512
    lr = 0.5
    for epoch in range(3):
        for i in range(0, len(x_train), bs):
            ypred = model(x_train[i:i+bs]) # have not been put through softmax etc.
            ytrue = y_train[i:i+bs]
            loss = F.cross_entropy(ypred, ytrue)
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= lr*p.grad
                    model.zero_grad() # p.grad.zero_() # could also do model.
            print(accuracy(ypred, ytrue), loss)
```

```{code-cell} ipython3
fit()
```

First, lets look at `__setattr__` and `__repr__`. Whenever you do `dc.a=` 
you are calling `__setattr__`.

```{code-cell} ipython3
class DummyClass:
    
    def __init__(self):
        self.stuff = dict() # this actually calls __setattr__
    
    def __setattr__(self,k,v):
        if k!= 'stuff':
            self.stuff[k] = v
        super().__setattr__(k,v)

    def __repr__(self): # must return a string
        return f'{self.stuff}'
        
        
```

```{code-cell} ipython3
dc = DummyClass()
dc # prints whatever is defined in __repr__
```

```{code-cell} ipython3
dc.a = 1
```

```{code-cell} ipython3
dc.b = 'chris'
```

```{code-cell} ipython3
dc
```

Behind the scenes, PyTorch overrides the `__setattr__` function in `nn.Module` so that the submodules you define are properly registered as parameters of the model.

```{code-cell} ipython3
class MyModule():
    def __init__(self, n_in, nh, n_out):
        self._modules = {}
        self.l1 = nn.Linear(n_in,nh)
        self.l2 = nn.Linear(nh,n_out)
        
    def __setattr__(self,k,v):
        if not k.startswith("_"): self._modules[k] = v
        super().__setattr__(k,v)
        
    def __repr__(self): return f'{self._modules}'
    
    def parameters(self):
        for l in self._modules.values():
            # TODO: rewrite with `yield from`
            for p in l.parameters(): yield p
```

```{code-cell} ipython3
mdl = MyModule(784,50,10)
mdl
```

```{code-cell} ipython3
for p in mdl.parameters():
    print(p.shape)
```

### Registering Modules

```{code-cell} ipython3
from functools import reduce
```

You can read more about `reduce` in the docs or this great [resource](https://realpython.com/python-reduce-function/).
the function (first arg) to `reduce` should take two arguments and then goes left to right
over the iterable in a cumulative fashion.

```{code-cell} ipython3
def my_add(a, b):
    result = a + b
    print(f"{a} + {b} = {result}")
    return result

numbers = [0, 1, 2, 3, 4]

reduce(my_add, numbers)
```

```{code-cell} ipython3
layers = [nn.Linear(784, 50), nn.ReLU(), nn.Linear(50, 10)]
```

Here we have to register the modules

```{code-cell} ipython3
class Model(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        for i,l in enumerate(self.layers):
            self.add_module(f'layer_{i}', l)
    
    def forward(self, x):
        return reduce(lambda val,layer: layer(val), self.layers, x)
```

```{code-cell} ipython3
model = Model(layers)
model
```

```{code-cell} ipython3
model(x_train).shape
```

### nn.ModuleList

`nn.ModuleList` does this for us.

```{code-cell} ipython3
class SequentialModel(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x
```

```{code-cell} ipython3
model = SequentialModel(layers)
model
```

```{code-cell} ipython3
fit()
```

```{code-cell} ipython3
F.cross_entropy(model(x_train), y_train), accuracy(model(x_train), y_train)
```

### nn.Sequential

`nn.Sequential` is a convenient class which does the same as the above:

```{code-cell} ipython3
model = nn.Sequential(nn.Linear(784,50), nn.ReLU() ,nn.Linear(50,10))
model
```

```{code-cell} ipython3
fit()
F.cross_entropy(model(x_train), y_train), accuracy(model(x_train), y_train)
```

### optim

looks something like this

```{code-cell} ipython3
class Optimizer():
    def __init__(self, params, lr=0.5):
        self.params = list(params)
        self.lr=lr
        
    def step(self):
        with torch.no_grad():
            for p in self.params:
                p -= p.grad * self.lr

    def zero_grad(self):
        for p in self.params:
            p.grad.data.zero_()
```

```{code-cell} ipython3
model = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,10))
opt = Optimizer(model.parameters())
```

```{code-cell} ipython3
def report(loss, preds, yb): print(f'{loss:.2f}, {accuracy(preds, yb):.2f}')
```

```{code-cell} ipython3
bs = 512
for epoch in range(10):
    for i in range(0, len(x_train), bs):
        x = x_train[i:i+bs]
        y = y_train[i:i+bs]
        ypred = model(x)
        loss = F.cross_entropy(ypred, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
    report(loss, ypred, y)
```

Now lets use the torch.optim

```{code-cell} ipython3
from torch import optim
```

```{code-cell} ipython3
model = nn.Sequential(nn.Linear(784, 50), nn.ReLU(), nn.Linear(50, 10))
opt = optim.SGD(model.parameters(), lr=0.5)
```

```{code-cell} ipython3
bs = 512
for epoch in range(10):
    for i in range(0, len(x_train), bs):
        x = x_train[i:i+bs]
        y = y_train[i:i+bs]
        ypred = model(x)
        loss = F.cross_entropy(ypred, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
    report(loss, ypred, y)
```

## Dataset and DataLoader

Previously, our loop iterated over batches (xb, yb) like this:

```python
for i in range(0, n, bs):
    xb,yb = train_ds[i:min(n,i+bs)]
    ...
```

Let's make our loop much cleaner, using a data loader:

```python
for xb,yb in train_dl:
    ...
```

[docs on datasets and dataloaders](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html)

Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.

```{code-cell} ipython3
class Dataset:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]
```

```{code-cell} ipython3
train_ds,valid_ds = Dataset(x_train, y_train),Dataset(x_valid, y_valid)
assert len(train_ds)==len(x_train)
assert len(valid_ds)==len(x_valid)
```

We can even get a batch, not just one item at once.

```{code-cell} ipython3
train_ds[0:5]
```

```{code-cell} ipython3
model = nn.Sequential(nn.Linear(784, 50), nn.ReLU(), nn.Linear(50, 10))
opt = optim.SGD(model.parameters(), lr=0.5)
```

```{code-cell} ipython3
bs = 512
for epoch in range(10):
    for i in range(0, len(train_ds), bs):
        x,y = train_ds[i:i+bs]
        ypred = model(x)
        loss = F.cross_entropy(ypred, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
    report(loss, ypred, y)
```

### DataLoader

Lets make it even better.
Check out this cool `__iter__`.
DataLoaders wrap a DataSet..

```{code-cell} ipython3
class DataLoader:
    def __init__(self, ds, bs):
        self.ds = ds
        self.bs = bs
        
    def __iter__(self):
        for i in range(0, len(self.ds), bs):
            yield self.ds[i:i+bs]
```

```{code-cell} ipython3
bs = 512
train_dl = DataLoader(train_ds, bs)
val_dl = DataLoader(valid_ds, bs)
```

```{code-cell} ipython3
next(iter(train_dl))
```

```{code-cell} ipython3
next(iter(val_dl))
```

```{code-cell} ipython3
xb = next(iter(train_ds))[0]
plt.imshow(xb.reshape(28,28))
```

```{code-cell} ipython3
model = nn.Sequential(nn.Linear(784, 50), nn.ReLU(), nn.Linear(50, 10))
opt = optim.SGD(model.parameters(), lr=0.5)
def fit():
    bs = 512
    for epoch in range(10):
        for x,y in train_dl:
            ypred = model(x)
            loss = F.cross_entropy(ypred, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
        report(loss, ypred, y)
fit()
```

### Random sampling

```{code-cell} ipython3
import random
```

```{code-cell} ipython3
class Sampler:
    
    def __init__(self, ds, shuffle=False):
        self.shuffle = shuffle
        self.n = len(ds)
    
    def __iter__(self):
        res = list(range(self.n))
        if self.shuffle:
            random.shuffle(res)
        return iter(res)
        
```

```{code-cell} ipython3
ss = Sampler(train_ds, shuffle=False)
```

```{code-cell} ipython3
it = iter(ss)
```

```{code-cell} ipython3
next(it)
```

```{code-cell} ipython3
next(it)
```

```{code-cell} ipython3
from itertools import islice
```

```{code-cell} ipython3
list(islice(ss, 5))
```

```{code-cell} ipython3
ss = Sampler(train_ds, shuffle=True)
list(islice(ss, 5))
```

```{code-cell} ipython3
import fastcore.all as fc
```

```{code-cell} ipython3
class BatchSampler():
    def __init__(self, sampler, bs, drop_last=False):
#         sampler,bs,drop_last = self.sampler,self.bs,self.drop_last
        fc.store_attr()

    def __iter__(self): yield from fc.chunked(iter(self.sampler), self.bs, drop_last=self.drop_last)
```

```{code-cell} ipython3
batchs = BatchSampler(ss, 4)
list(islice(batchs, 5))
```

```{code-cell} ipython3
def collate(b):
    xs,ys = zip(*b)
    return torch.stack(xs),torch.stack(ys)
```

```{code-cell} ipython3
collate([train_ds[:5], train_ds[5:10]])
```

```{code-cell} ipython3
collate([train_ds[:5], train_ds[5:10]])[0].shape
```

```{code-cell} ipython3
collate([train_ds[:5], train_ds[5:10]])[1].shape
```

```{code-cell} ipython3
class DataLoader():
    
    def __init__(self, ds, batchs, collate_fn=collate):
        fc.store_attr()
        
    def __iter__(self):
        for b in self.batchs:
            yield self.collate_fn(self.ds[i] for i in b)
```

```{code-cell} ipython3
train_samp = BatchSampler(Sampler(train_ds, shuffle=True ), bs)
valid_samp = BatchSampler(Sampler(valid_ds, shuffle=False), bs)
```

```{code-cell} ipython3
train_dl = DataLoader(train_ds, batchs=train_samp, collate_fn=collate)
valid_dl = DataLoader(valid_ds, batchs=valid_samp, collate_fn=collate)
```

```{code-cell} ipython3
xb,yb = next(iter(valid_dl))
plt.imshow(xb[0].view(28,28))
yb[0]
```

```{code-cell} ipython3
xb.shape, yb.shape
```

```{code-cell} ipython3
model = nn.Sequential(nn.Linear(784, 50), nn.ReLU(), nn.Linear(50, 10))
opt = optim.SGD(model.parameters(), lr=0.5)
fit()
```

### PyTorch DataLoader
- also takes care of multi processing (doing stuff in parallel). Similar to TF.

```{code-cell} ipython3
train_ds[0]
```

```{code-cell} ipython3
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, BatchSampler
```

```{code-cell} ipython3
bs = 512
train_samp = BatchSampler(RandomSampler(train_ds),     bs, drop_last=False)
valid_samp = BatchSampler(SequentialSampler(valid_ds), bs, drop_last=False)
```

```{code-cell} ipython3
train_dl = DataLoader(train_ds, batch_sampler=train_samp, collate_fn=collate)
valid_dl = DataLoader(valid_ds, batch_sampler=valid_samp, collate_fn=collate)
```

```{code-cell} ipython3
next(iter(train_dl))
```

```{code-cell} ipython3
model = nn.Sequential(nn.Linear(784, 50), nn.ReLU(), nn.Linear(50, 10))
opt = optim.SGD(model.parameters(), lr=0.5)
fit()
```

PyTorch can auto-generate the BatchSampler for us:

```{code-cell} ipython3
train_dl = DataLoader(train_ds, bs, sampler=RandomSampler(train_ds), collate_fn=collate)
valid_dl = DataLoader(valid_ds, bs, sampler=SequentialSampler(valid_ds), collate_fn=collate)
```

PyTorch can also generate the Sequential/RandomSamplers too:

```{code-cell} ipython3
train_dl = DataLoader(train_ds, bs, shuffle=True, drop_last=True, num_workers=2)
valid_dl = DataLoader(valid_ds, bs, shuffle=False, num_workers=2)
```

```{code-cell} ipython3
model = nn.Sequential(nn.Linear(784, 50), nn.ReLU(), nn.Linear(50, 10))
opt = optim.SGD(model.parameters(), lr=0.5)
fit()
```

```{code-cell} ipython3
train_ds[[4,6,7]]
```

```{code-cell} ipython3
train_dl = DataLoader(train_ds, sampler=train_samp)
valid_dl = DataLoader(valid_ds, sampler=valid_samp)
```

```{code-cell} ipython3
for x,y in train_dl:
    break
y.shape
```

```{code-cell} ipython3
xb,yb = next(iter(train_dl))
xb.shape, yb.shape
```

## Validation

+++

You **always** should also have a [validation set](http://www.fast.ai/2017/11/13/validation-sets/), in order to identify if you are overfitting.

We will calculate and print the validation loss at the end of each epoch.

(Note that we always call `model.train()` before training, and `model.eval()` before inference, because these are used by layers such as `nn.BatchNorm2d` and `nn.Dropout` to ensure appropriate behaviour for these different phases.)

```{code-cell} ipython3
def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb,yb in train_dl:
            loss = loss_func(model(xb), yb)
            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
        with torch.no_grad():
            tot_loss,tot_acc,count = 0.,0.,0
            for xb,yb in valid_dl:
                pred = model(xb)
                n = len(xb)
                count += n
                tot_loss += loss_func(pred,yb).item()*n
                tot_acc  += accuracy (pred,yb).item()*n
        print(epoch, tot_loss/count, tot_acc/count)
    return tot_loss/count, tot_acc/count
```

```{code-cell} ipython3
model = nn.Sequential(nn.Linear(784, 50), nn.ReLU(), nn.Linear(50, 10))
opt = optim.SGD(model.parameters(), lr=0.5)
fit(10, model, F.cross_entropy, opt, train_dl, val_dl)
```

Some good [docs](https://pytorch.org/docs/stable/data.html) on dataloaders in general.

Good [blog](https://www.scottcondron.com/jupyter/visualisation/audio/2020/12/02/dataloaders-samplers-collate.html) on datasets and dataloaders etc.

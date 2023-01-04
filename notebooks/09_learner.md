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
#| default_exp learner
```

```{code-cell} ipython3
#|export
import pickle,gzip,math,os,time,shutil,torch,matplotlib as mpl,numpy as np,matplotlib.pyplot as plt
import fastcore.all as fc
from collections.abc import Mapping
from pathlib import Path
from operator import attrgetter,itemgetter
from functools import partial
from copy import copy
from contextlib import contextmanager
from warnings import warn

from torch import tensor,nn,optim
from torch.utils.data import DataLoader,default_collate
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from datasets import load_dataset,load_dataset_builder

from miniai.datasets import *
from miniai.conv import *

from fastprogress import progress_bar,master_bar
```

```{code-cell} ipython3
from fastcore.test import test_close

torch.set_printoptions(precision=2, linewidth=140, sci_mode=False)
torch.manual_seed(1)
mpl.rcParams['image.cmap'] = 'gray'

import logging
logging.disable(logging.WARNING)
```

## Learner

```{code-cell} ipython3
x,y = 'image','label'
name = "fashion_mnist"
dsd = load_dataset(name)
```

```{code-cell} ipython3
@inplace
def transformi(b): b[x] = [torch.flatten(TF.to_tensor(o)) for o in b[x]]
```

```{code-cell} ipython3
bs = 1024
tds = dsd.with_transform(transformi)
```

```{code-cell} ipython3
#|export
class DataLoaders:
    def __init__(self, *dls): self.train,self.valid = dls[:2]

    @classmethod
    def from_dd(cls, dd, batch_size, as_tuple=True, **kwargs):
        return cls(*[DataLoader(ds, batch_size, collate_fn=collate_dict(ds), **kwargs) for ds in dd.values()])
```

```{code-cell} ipython3
dls = DataLoaders.from_dd(tds, bs, num_workers=4)
dt = dls.train
xb,yb = next(iter(dt))
xb.shape,yb[:10]
```

Basic learner to show the main idea but we will be making it better down below
so it can handle more cases than just accuracies etc.

```{code-cell} ipython3
class Learner:
    def __init__(self, model, dls, loss_func, lr, opt_func=optim.SGD): fc.store_attr()

    def one_batch(self):
        self.xb,self.yb = to_device(self.batch)
        self.preds = self.model(self.xb)
        self.loss = self.loss_func(self.preds, self.yb)
        if self.model.training:
            self.loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        with torch.no_grad(): self.calc_stats()

    def calc_stats(self):
        acc = (self.preds.argmax(dim=1)==self.yb).float().sum()
        self.accs.append(acc)
        n = len(self.xb)
        self.losses.append(self.loss*n)
        self.ns.append(n)

    def one_epoch(self, train):
        self.model.training = train
        dl = self.dls.train if train else self.dls.valid
        for self.num,self.batch in enumerate(dl): self.one_batch()
        n = sum(self.ns)
        print(self.epoch, self.model.training, sum(self.losses).item()/n, sum(self.accs).item()/n)
    
    def fit(self, n_epochs):
        self.accs,self.losses,self.ns = [],[],[]
        self.model.to(def_device)
        self.opt = self.opt_func(self.model.parameters(), self.lr)
        self.n_epochs = n_epochs
        for self.epoch in range(n_epochs):
            self.one_epoch(True)
            with torch.no_grad(): self.one_epoch(False)
```

```{code-cell} ipython3
m,nh = 28*28,50
model = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,10))
```

```{code-cell} ipython3
learn = Learner(model, dls, F.cross_entropy, lr=0.2)
learn.fit(1)
```

## Basic Callbacks Learner

```{code-cell} ipython3
#|export
class CancelFitException(Exception): pass
class CancelBatchException(Exception): pass
class CancelEpochException(Exception): pass
```

```{code-cell} ipython3
#|export
class Callback(): order = 0
```

```{code-cell} ipython3
#|export
def run_cbs(cbs, method_nm, learn=None):
    for cb in sorted(cbs, key=attrgetter('order')):
        method = getattr(cb, method_nm, None)
        if method is not None: method(learn)
```

Each call back subclasses `Callback` which just gives the order.
I think the methods like `before_fit, before_epoch, after_fit`, etc must take a `learn` argument.
If we don't need it then leave it as `None`.

```{code-cell} ipython3
class CompletionCB(Callback):
    def before_fit(self, learn):
        self.count = 0
    def after_batch(self, learn):
        self.count += 1
    def after_fit(self, learn):
        print(f'Completed {self.count} batches')
```

```{code-cell} ipython3
cb = CompletionCB()
cb.before_fit(None)
print(cb.count)
cb.after_batch(None)
cb.after_batch(None)
cb.after_batch(None)
cb.after_fit(None)
```

Start with a list of callbacks.
Lets see what the function `run_cbs` does.

```{code-cell} ipython3
class FakeCB1(Callback):
    order = 1
    
    def before_fit(self, learn=None):
        print(f'I have order {self.order}')

class FakeCB2(Callback):
    order = 2
    
    def before_fit(self, learn=None):
        print(f'I have order {self.order}')
        self.count = 0
    
    def after_batch(self, learn):
        self.count += 1
    
class FakeCB3(Callback):
    order = 3
    
    def before_fit(self, learn=None):
        print(f'I have order {self.order}')    
```

```{code-cell} ipython3
cbs = [FakeCB3(), FakeCB1(), FakeCB2()]
```

This line of code in the `run_cbs` function sorts the callbacks by the `order`
attribute. Cool use of `from operator import attrgetter`

```{code-cell} ipython3
for cb in sorted(cbs, key=attrgetter('order')): # from operator import attrgetter,itemgetter
    print(type(cb).__name__)
```

Once the call backs list is sorted and we have a specific call back we use 
`getattr` to grab the method/function from that specific class call back.
`getattr` is built into base Python. 
Get a named attribute from an object; `getattr(x, 'y')` is equivalent to `x.y`.

```{code-cell} ipython3
cb = FakeCB2()
method = getattr(cb, 'not_there')
```

```{code-cell} ipython3
getattr(cb, 'not_there', None) is None
```

```{code-cell} ipython3
method = getattr(cb, 'before_fit', None)
method
```

```{code-cell} ipython3
method(None)
cb.count
```

```{code-cell} ipython3
method = getattr(cb, 'after_batch', None) # Remember, this is not calling the method. Just defining it
method(None)
cb.count
```

```{code-cell} ipython3
cbs = [FakeCB1(), FakeCB3(),  FakeCB2(),]
run_cbs(cbs, 'before_fit')
run_cbs(cbs, 'after_batch')
run_cbs(cbs, 'after_fit')
```

```{code-cell} ipython3
cbs = [CompletionCB()]
run_cbs(cbs, 'before_fit')
run_cbs(cbs, 'after_batch')
run_cbs(cbs, 'after_fit')
```

Slightly more sophisticated learner with callbacks argument `cbs`.
Notice we took the hard coded accuracy metrics out. Took out the `calc_stats`.
Have some try catch stuff now to with call backs etc.
Note that we are passing the learner object `self` to `run_cbs` in side this logic:

`def callback(self, method_nm): run_cbs(self.cbs, method_nm, self)`

```{code-cell} ipython3
class Learner():
    def __init__(self, model, dls, loss_func, lr, cbs, opt_func=optim.SGD): fc.store_attr()

    def one_batch(self):
        self.preds = self.model(self.batch[0])
        self.loss = self.loss_func(self.preds, self.batch[1])
        if self.model.training:
            self.loss.backward()
            self.opt.step()
            self.opt.zero_grad()

    def one_epoch(self, train):
        self.model.train(train)
        self.dl = self.dls.train if train else self.dls.valid
        try:
            self.callback('before_epoch')
            for self.iter,self.batch in enumerate(self.dl):
                try:
                    self.callback('before_batch')
                    self.one_batch()
                    self.callback('after_batch')
                except CancelBatchException: pass
            self.callback('after_epoch')
        except CancelEpochException: pass
    
    def fit(self, n_epochs):
        self.n_epochs = n_epochs
        self.epochs = range(n_epochs)
        self.opt = self.opt_func(self.model.parameters(), self.lr)
        try:
            self.callback('before_fit')
            for self.epoch in self.epochs:
                self.one_epoch(True)
                self.one_epoch(False)
            self.callback('after_fit')
        except CancelFitException: pass

    def callback(self, method_nm): run_cbs(self.cbs, method_nm, self)
```

```{code-cell} ipython3
m,nh = 28*28,50
def get_model(): return nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,10))
```

```{code-cell} ipython3
model = get_model()
learn = Learner(model, dls, F.cross_entropy, lr=0.2, cbs=[CompletionCB()])
learn.fit(2)
```

If we wanted to log the epochs for example too

```{code-cell} ipython3
class CompletionCB(Callback):
    def before_fit(self, learn):
        self.count = 0
    def after_batch(self, learn):
        self.count += 1
    def after_fit(self, learn):
        print(f'Completed {self.count} batches')
    def before_epoch(self, learn):
        print(f'Starting epoch {learn.epoch} with training mode {learn.model.training}')
    def after_epoch(self, learn):
        print(f'Finished epoch {learn.epoch} with training mode {learn.model.training}')
```

That's why it is so nice to pass the `learn` object to the callback methods.
We get to access all the `learn` attributes.

```{code-cell} ipython3
model = get_model()
learn = Learner(model, dls, F.cross_entropy, lr=0.2, cbs=[CompletionCB()])
learn.fit(3)
```

Notice at this point model training is super slow!

```{code-cell} ipython3
learn.model
```

```{code-cell} ipython3
next(learn.model.parameters()).is_cuda
```

The model is not on the GPU.

```{code-cell} ipython3
for xx,yy in dls.train: # we already used the name x and y above duh!
    break
print(xx.shape, yy.shape)
```

```{code-cell} ipython3
xx.is_cuda, yy.is_cuda
```

Neither is the data on GPU.

+++

Here is a callback to just run a single batch.
Useful for debugging things or running things for one batch
to visualize something etc.

```{code-cell} ipython3
#| export
class SingleBatchCB(Callback):
    order = 1
    def after_batch(self, learn): raise CancelFitException()
```

```{code-cell} ipython3
learn = Learner(get_model(), dls, F.cross_entropy, lr=0.2, cbs=[SingleBatchCB(), CompletionCB()])
learn.fit(5)
```

## Metrics

```{code-cell} ipython3
class Metric:
    def __init__(self):
        self.reset()
    def reset(self):
        self.vals,self.ns = [],[]
        
    def add(self, inp, targ=None, n=1):
        self.last = self.calc(inp, targ)
        self.vals.append(self.last)
        self.ns.append(n)
    
    @property
    def value(self):
        ns = tensor(self.ns)
        return (tensor(self.vals)*ns).sum()/ns.sum()
    
    def calc(self, inps, targs): return inps
```

```{code-cell} ipython3
class Accuracy(Metric):
    def calc(self, inps, targs):
        return (inps==targs).float().mean()
```

```{code-cell} ipython3
acc = Accuracy()
acc.add(tensor([0, 1, 2, 0, 1, 2]), tensor([0, 1, 1, 2, 1, 0]))
acc.add(tensor([1, 1, 2, 0, 1]), tensor([0, 1, 1, 2, 1]))
acc.value
```

```{code-cell} ipython3
loss = Metric()
loss.add(0.6, n=32)
loss.add(0.9, n=2)
loss.value, round((0.6*32+0.9*2)/(32+2), 2)
```

## Some callbacks

+++

```
pip install torcheval
```

```{code-cell} ipython3
#|export
from torcheval.metrics import MulticlassAccuracy,Mean
```

```{code-cell} ipython3
metric = MulticlassAccuracy()
metric.update(tensor([0, 2, 1, 3]), tensor([0, 1, 2, 3]))
metric.compute()
```

```{code-cell} ipython3
metric.reset()
metric.compute()
```

```{code-cell} ipython3
#|export
def to_cpu(x):
    if isinstance(x, Mapping): return {k:to_cpu(v) for k,v in x.items()}
    if isinstance(x, list): return [to_cpu(o) for o in x]
    if isinstance(x, tuple): return tuple(to_cpu(list(x)))
    return x.detach().cpu()
```

```{code-cell} ipython3
#|export
class MetricsCB(Callback):
    def __init__(self, *ms, **metrics):
        for o in ms: metrics[type(o).__name__] = o
        self.metrics = metrics
        self.all_metrics = copy(metrics)
        self.all_metrics['loss'] = self.loss = Mean()

    def _log(self, d): print(d)
    def before_fit(self, learn): learn.metrics = self
    def before_epoch(self, learn): [o.reset() for o in self.all_metrics.values()]

    def after_epoch(self, learn):
        log = {k:f'{v.compute():.3f}' for k,v in self.all_metrics.items()}
        log['epoch'] = learn.epoch
        log['train'] = 'train' if learn.model.training else 'eval'
        self._log(log)

    def after_batch(self, learn):
        x,y,*_ = to_cpu(learn.batch)
        for m in self.metrics.values(): m.update(to_cpu(learn.preds), y)
        self.loss.update(to_cpu(learn.loss), weight=len(x))
```

```{code-cell} ipython3
#|export
class DeviceCB(Callback):
    def __init__(self, device=def_device): fc.store_attr()
    def before_fit(self, learn): learn.model.to(self.device)
    def before_batch(self, learn): learn.batch = to_device(learn.batch, device=self.device)
```

```{code-cell} ipython3
model = get_model()
metrics = MetricsCB(accuracy=MulticlassAccuracy())
learn = Learner(model, dls, F.cross_entropy, lr=0.2, cbs=[DeviceCB(), metrics])
learn.fit(10)
```

The above is using some of GPU but not much! It says like 1% GPU usage.

+++

## Flexible learner

```{code-cell} ipython3
#|export
class Learner():
    def __init__(self, model, dls=(0,), loss_func=F.mse_loss, lr=0.1, cbs=None, opt_func=optim.SGD):
        cbs = fc.L(cbs)
        fc.store_attr()

    @contextmanager
    def cb_ctx(self, nm):
        try:
            self.callback(f'before_{nm}')
            yield
            self.callback(f'after_{nm}')
        except globals()[f'Cancel{nm.title()}Exception']: pass
        finally: self.callback(f'cleanup_{nm}')
                
    def one_epoch(self, train):
        self.model.train(train)
        self.dl = self.dls.train if train else self.dls.valid
        with self.cb_ctx('epoch'):
            for self.iter,self.batch in enumerate(self.dl):
                with self.cb_ctx('batch'):
                    self.predict()
                    self.get_loss()
                    if self.training:
                        self.backward()
                        self.step()
                        self.zero_grad()
    
    def fit(self, n_epochs=1, train=True, valid=True, cbs=None, lr=None):
        cbs = fc.L(cbs)
        # `add_cb` and `rm_cb` were added in lesson 18
        for cb in cbs: self.cbs.append(cb)
        try:
            self.n_epochs = n_epochs
            self.epochs = range(n_epochs)
            self.opt = self.opt_func(self.model.parameters(), self.lr if lr is None else lr)
            with self.cb_ctx('fit'):
                for self.epoch in self.epochs:
                    if train: self.one_epoch(True)
                    if valid: torch.no_grad()(self.one_epoch)(False)
        finally:
            for cb in cbs: self.cbs.remove(cb)

    def __getattr__(self, name):
        if name in ('predict','get_loss','backward','step','zero_grad'): return partial(self.callback, name)
        raise AttributeError(name)

    def callback(self, method_nm): run_cbs(self.cbs, method_nm, self)
    
    @property
    def training(self): return self.model.training
```

Interesting so `('predict','get_loss','backward','step','zero_grad')` functions are implemented as callbacks.
I guess for flexibility. See `__getattr__` above. 

```{code-cell} ipython3
#|export
class TrainCB(Callback):
    def predict(self, learn): learn.preds = learn.model(learn.batch[0])
    def get_loss(self, learn): learn.loss = learn.loss_func(learn.preds, learn.batch[1])
    def backward(self, learn): learn.loss.backward()
    def step(self, learn): learn.opt.step()
    def zero_grad(self, learn): learn.opt.zero_grad()
```

```{code-cell} ipython3
#|export
class ProgressCB(Callback):
    order = MetricsCB.order+1
    def __init__(self, plot=False): self.plot = plot
    def before_fit(self, learn):
        learn.epochs = self.mbar = master_bar(learn.epochs)
        self.first = True
        if hasattr(learn, 'metrics'): learn.metrics._log = self._log
        self.losses = []

    def _log(self, d):
        if self.first:
            self.mbar.write(list(d), table=True)
            self.first = False
        self.mbar.write(list(d.values()), table=True)

    def before_epoch(self, learn): learn.dl = progress_bar(learn.dl, leave=False, parent=self.mbar)
    def after_batch(self, learn):
        learn.dl.comment = f'{learn.loss:.3f}'
        if self.plot and hasattr(learn, 'metrics') and learn.training:
            self.losses.append(learn.loss.item())
            self.mbar.update_graph([[fc.L.range(self.losses), self.losses]])
```

```{code-cell} ipython3
model = get_model()
```

```{code-cell} ipython3
metrics = MetricsCB(accuracy=MulticlassAccuracy())
cbs = [TrainCB(), DeviceCB(), metrics, ProgressCB(plot=True)]
learn = Learner(model, dls, F.cross_entropy, lr=0.2, cbs=cbs)
learn.fit(3)
```

## TrainLearner and MomentumLearner

+++

I guess here we made a design decision change. Note that we are not
using `Callback` here anymore. We are subclassing  `Learner` class
and not passing `learn` to all the methods now. So now the learning
is not done through a call back but actually by subclassing the Learner class.

```{code-cell} ipython3
#|export
class TrainLearner(Learner):
    def predict(self): self.preds = self.model(self.batch[0])
    def get_loss(self): self.loss = self.loss_func(self.preds, self.batch[1])
    def backward(self): self.loss.backward()
    def step(self): self.opt.step()
    def zero_grad(self): self.opt.zero_grad()
```

```{code-cell} ipython3
#|export
class MomentumLearner(TrainLearner):
    def __init__(self, model, dls, loss_func, lr=None, cbs=None, opt_func=optim.SGD, mom=0.85):
        self.mom = mom
        super().__init__(model, dls, loss_func, lr, cbs, opt_func)

    def zero_grad(self):
        with torch.no_grad():
            for p in self.model.parameters(): p.grad *= self.mom
```

This is why we dont use the train call back below.

```{code-cell} ipython3
# NB: No TrainCB
metrics = MetricsCB(accuracy=MulticlassAccuracy())
cbs = [DeviceCB(), metrics, ProgressCB(plot=True)]
learn = MomentumLearner(get_model(), dls, F.cross_entropy, lr=0.2, cbs=cbs)
learn.fit(5)
```

## LRFinderCB

```{code-cell} ipython3
class LRFinderCB(Callback):
    def __init__(self, lr_mult=1.3): fc.store_attr()
    
    def before_fit(self, learn):
        self.lrs,self.losses = [],[]
        self.min = math.inf

    def after_batch(self, learn):
        if not learn.training: raise CancelEpochException()
        self.lrs.append(learn.opt.param_groups[0]['lr'])
        loss = to_cpu(learn.loss)
        self.losses.append(loss)
        if loss < self.min: self.min = loss
        if loss > self.min*3: raise CancelFitException()
        for g in learn.opt.param_groups: g['lr'] *= self.lr_mult
```

```{code-cell} ipython3
lrfind = LRFinderCB()
cbs = [DeviceCB(), lrfind]
learn = MomentumLearner(get_model(), dls, F.cross_entropy, lr=1e-4, cbs=cbs)
learn.fit(4)
plt.plot(lrfind.lrs, lrfind.losses)
plt.xscale('log')
```

```{code-cell} ipython3
#|export
from torch.optim.lr_scheduler import ExponentialLR
```

[ExponentialLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR)

```{code-cell} ipython3
#|export
class LRFinderCB(Callback):
    def __init__(self, gamma=1.3, max_mult=3): fc.store_attr()
    
    def before_fit(self, learn):
        self.sched = ExponentialLR(learn.opt, self.gamma)
        self.lrs,self.losses = [],[]
        self.min = math.inf

    def after_batch(self, learn):
        if not learn.training: raise CancelEpochException()
        self.lrs.append(learn.opt.param_groups[0]['lr'])
        loss = to_cpu(learn.loss)
        self.losses.append(loss)
        if loss < self.min: self.min = loss
        if loss > self.min*self.max_mult:
            raise CancelFitException()
        self.sched.step()

    def cleanup_fit(self, learn):
        plt.plot(self.lrs, self.losses)
        plt.xscale('log')
```

```{code-cell} ipython3
cbs = [DeviceCB()]
learn = MomentumLearner(get_model(), dls, F.cross_entropy, lr=1e-5, cbs=cbs)
learn.fit(3, cbs=LRFinderCB())
```

```{code-cell} ipython3
#|export
@fc.patch
def lr_find(self:Learner, gamma=1.3, max_mult=3, start_lr=1e-5, max_epochs=10):
    self.fit(max_epochs, lr=start_lr, cbs=LRFinderCB(gamma=gamma, max_mult=max_mult))
```

`lr_find` was added in lesson 18. It's just a shorter way of using `LRFinderCB`.

```{code-cell} ipython3
MomentumLearner(get_model(), dls, F.cross_entropy, cbs=cbs).lr_find()
```

## Updated versions since the lesson

+++

After the lesson we noticed that `contextlib.context_manager` has a surprising "feature" which doesn't let us raise an exception before the `yield`. Therefore we've replaced the context manager in this updated version of `Learner`, and have also added a few more callbacks in `one_epoch()`:

```{code-cell} ipython3
#|export
class _CbCtxInner:
    def __init__(self, outer, nm): self.outer,self.nm = outer,nm
    def __enter__(self): self.outer.callback(f'before_{self.nm}')
    def __exit__ (self, exc_type, exc_val, traceback):
        chk_exc = globals()[f'Cancel{self.nm.title()}Exception']
        try:
            if not exc_type: self.outer.callback(f'after_{self.nm}')
            return exc_type==chk_exc
        except chk_exc: pass
        finally: self.outer.callback(f'cleanup_{self.nm}')
```

```{code-cell} ipython3
#|export
class Learner():
    def __init__(self, model, dls=(0,), loss_func=F.mse_loss, lr=0.1, cbs=None, opt_func=optim.SGD):
        cbs = fc.L(cbs)
        fc.store_attr()

    def cb_ctx(self, nm): return _CbCtxInner(self, nm)
                
    def one_epoch(self, train):
        self.model.train(train)
        self.dl = self.dls.train if train else self.dls.valid
        with self.cb_ctx('epoch'):
            for self.iter,self.batch in enumerate(self.dl):
                with self.cb_ctx('batch'):
                    self.predict()
                    self.callback('after_predict')
                    self.get_loss()
                    self.callback('after_loss')
                    if self.training:
                        self.backward()
                        self.callback('after_backward')
                        self.step()
                        self.callback('after_step')
                        self.zero_grad()
    
    def fit(self, n_epochs=1, train=True, valid=True, cbs=None, lr=None):
        cbs = fc.L(cbs)
        # `add_cb` and `rm_cb` were added in lesson 18
        for cb in cbs: self.cbs.append(cb)
        try:
            self.n_epochs = n_epochs
            self.epochs = range(n_epochs)
            self.opt = self.opt_func(self.model.parameters(), self.lr if lr is None else lr)
            with self.cb_ctx('fit'):
                for self.epoch in self.epochs:
                    if train: self.one_epoch(True)
                    if valid: torch.no_grad()(self.one_epoch)(False)
        finally:
            for cb in cbs: self.cbs.remove(cb)

    def __getattr__(self, name):
        if name in ('predict','get_loss','backward','step','zero_grad'): return partial(self.callback, name)
        raise AttributeError(name)

    def callback(self, method_nm): run_cbs(self.cbs, method_nm, self)
    
    @property
    def training(self): return self.model.training
```

```{code-cell} ipython3
model = get_model()

metrics = MetricsCB(accuracy=MulticlassAccuracy())
cbs = [TrainCB(), DeviceCB(), metrics, ProgressCB(plot=True)]
learn = Learner(model, dls, F.cross_entropy, lr=0.2, cbs=cbs)
learn.fit(1)
```

## Export -

```{code-cell} ipython3
import nbdev; nbdev.nbdev_export()
```

```{code-cell} ipython3

```

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
```

- to get the miniai library below to behave properly
need to `pip install -e .` within the `/workspace/course22p2` dir

```{code-cell} ipython3
os.chdir('course22p2/')
os.system('pip install -e .')
os.chdir('/workspace')
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

```{code-cell} ipython3
xb,yb = next(iter(dt))
xb.shape,yb[:10]
```

Lets first look at a basic learner class:

```{code-cell} ipython3
#|export
class Learner:
    def __init__(self, model, dls, loss_func, lr, opt_func=optim.SGD):
        fc.store_attr() 

    def one_batch(self):
        self.xb,self.yb = to_device(self.batch)
        self.preds = self.model(self.xb)
        self.loss = self.loss_func(self.preds, self.yb)
        if self.model.training:
            self.loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        with torch.no_grad():
            self.calc_stats()
    
    def calc_stats(self):
        acc = (self.preds.argmax(dim=1)==self.yb).float().sum()
        self.accs.append(acc)
        n = len(self.xb)
        self.losses.append(self.loss*n)
        self.ns.append(n)

    def one_epoch(self, train):
        self.model.training = train
        dl = self.dls.train if train else self.dls.valid
        for self.num, self.batch in enumerate(dl):
            self.one_batch()
        n = sum(self.ns)
        print(self.epoch, self.model.training, sum(self.losses).item()/n, sum(self.accs).item()/n)
    
    def fit(self, n_epochs):
        self.accs,self.losses,self.ns = [],[],[]
        self.model.to(def_device)
        self.opt = self.opt_func(self.model.parameters(), self.lr)
        self.n_epochs = n_epochs
        for self.epoch in range(n_epochs):
            self.one_epoch(True)
            self.one_epoch(False)
```

```{code-cell} ipython3
m,nh = 28*28,50
model = nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,10))
```

```{code-cell} ipython3
self = Learner(model, dls, F.cross_entropy, lr=0.2)
```

```{code-cell} ipython3
self.fit(5)
```

```{code-cell} ipython3
self.model.training
```

## Basic Callbacks Learner

```{code-cell} ipython3
class CancelFitException(Exception): pass
class CancelBatchException(Exception): pass
class CancelEpochException(Exception): pass
```

```{code-cell} ipython3
def run_cbs(cbs, method_nm):
    for cb in sorted(cbs, key = attrgetter('order')):
        method = getattr(cb, method_nm, None)
        if method is not None:
            method()
```

```{code-cell} ipython3
class FakeCallback1():
    order = 1
    def print_me(self):
        print('Hello1')
    
class FakeCallback2():
    order = 0
    def print_me(self):
        print('Hello2')

class FakeCallback3():
    order = 2
    def print_me(self):
        print('Hello3')
```

Okay so just to see what this `run_cbs` does:

```{code-cell} ipython3
run_cbs([FakeCallback3(), FakeCallback1(), FakeCallback2()], 'print_me')
```

```{code-cell} ipython3
#|export
class Callback():
    order = 0
```

```{code-cell} ipython3
class CompletionCB(Callback):
    def before_fit(self):
        self.count = 0
    def after_batch(self):
        self.count += 1
    def after_fit(self):
        print(f'Completed {self.count} batches')
```

```{code-cell} ipython3
cbs = [CompletionCB()]
```

```{code-cell} ipython3
run_cbs(cbs, 'before_fit') # so loop over the cbs and call this method name for each cb if its there
```

```{code-cell} ipython3
run_cbs(cbs, 'after_batch')
```

```{code-cell} ipython3
run_cbs(cbs, 'after_fit')
```

```{code-cell} ipython3
run_cbs(cbs, 'after_batch')
run_cbs(cbs, 'after_batch')
run_cbs(cbs, 'after_batch')
run_cbs(cbs, 'after_fit')
```

```{code-cell} ipython3
class Learner():
    def __init__(self, model, dls, loss_func,lr, cbs, opt_func=optim.SGD):
        fc.store_attr()
        for cb in cbs:
            # so each call back gets the attribute learn
            # which is an instance of this Learner class
            cb.learn = self

    def one_batch(self):
        xb, yb = self.batch
        self.preds = self.model(xb)
        self.loss = self.loss_func(self.preds, yb)
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

    def callback(self, method_nm): run_cbs(self.cbs, method_nm)
```

```{code-cell} ipython3
m,nh = 28*28,50
def get_model(): 
    return nn.Sequential(nn.Linear(m,nh), nn.ReLU(), nn.Linear(nh,10))
```

```{code-cell} ipython3
model = get_model()
learn = Learner(model, dls, F.cross_entropy, lr=0.2, cbs=[CompletionCB()])
learn.fit(1)
```

```{code-cell} ipython3
class SingleBatchCB(Callback):
    order = 1
    def after_batch(self):
        raise CancelEpochException()
```

```{code-cell} ipython3
learn = Learner(get_model(), dls, F.cross_entropy, lr=0.2, cbs=[SingleBatchCB(), CompletionCB()])
learn.fit(1)
```

## Metrics

```{code-cell} ipython3
class Metric:
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.vals, self.ns = [], []
        
    def add(self, inp, targ=None, n=1):
        self.last = self.calc(inp, targ)
        self.vals.append(self.last)
        self.ns.append(n)
    
    @property
    def value(self):
        ns = tensor(self.ns)
        return (tensor(self.vals)*ns).sum()/ns.sum()
    
    def calc(self, inps, targs):
        return inps
```

```{code-cell} ipython3
class Accuracy(Metric):
    def calc(self, inps, targs):
        return (inps==targs).float().mean()
```

```{code-cell} ipython3
acc = Accuracy()
acc.add(tensor([0, 1, 2, 0, 1, 2]), tensor([0, 1, 2, 0, 1, 2]))
acc.add(tensor([1, 1, 2, 0, 1]), tensor([0, 1, 1, 2, 1]))
acc.value
```

```{code-cell} ipython3
loss = Metric()
loss.add(0.6, n=32)
loss.add(0.9, n=2)
loss.value, round((0.6*32+0.9*2)/(32+2), 2)
```

```{code-cell} ipython3
#|export
# put stuff on proper device before training
class DeviceCB(Callback):
    def __init__(self, device=def_device):
        fc.store_attr()
    def before_fit(self):
        self.learn.model.to(self.device)
    def before_batch(self):
        self.learn.batch = to_device(self.learn.batch, device=self.device)
```

```{code-cell} ipython3
os.system('pip install torcheval')
```

```{code-cell} ipython3
#|export
from torcheval.metrics import MulticlassAccuracy, Mean
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
        for o in ms:
            metrics[type(o).__name__] = o
        self.metrics = metrics
        self.all_metrics = copy(metrics)
        self.all_metrics['loss'] = self.loss = Mean()

    def _log(self, d):
        print(d)
        
    def before_fit(self):
        self.learn.metrics = self
        
    def before_epoch(self):
        [o.reset() for o in self.all_metrics.values()]
        
    def after_epoch(self):
        log = {k:f'{v.compute():.3f}' for k,v in self.all_metrics.items()}
        log['epoch'] = self.learn.epoch
        log['train'] = self.learn.model.training
        self._log(log)

    def after_batch(self):
        x,y = to_cpu(self.learn.batch)
        for m in self.metrics.values():
            m.update(to_cpu(self.learn.preds), y)
        self.loss.update(to_cpu(self.learn.loss), weight=len(x))
```

```{code-cell} ipython3
model = get_model()
metrics = MetricsCB(accuracy=MulticlassAccuracy())
learn = Learner(model, dls, F.cross_entropy, lr=0.2, cbs=[DeviceCB(), metrics])
learn.fit(1)
```

## Flexible learner

```{code-cell} ipython3
#|export
class Learner():
    def __init__(self, model, dls, loss_func, lr, cbs, opt_func=optim.SGD):
        fc.store_attr()
        for cb in cbs:
            cb.learn = self

    @contextmanager
    def callback_ctx(self, nm):
        try:
            self.callback(f'before_{nm}')
            yield # context manager
        except globals()[f'Cancel{nm.title()}Exception']:
            pass
        finally:
            self.callback(f'after_{nm}')

    def one_epoch(self, train):
        self.model.train(train)
        self.dl = self.dls.train if train else self.dls.valid
        with self.callback_ctx('epoch'):
            for self.iter,self.batch in enumerate(self.dl):
                with self.callback_ctx('batch'):
                    self.predict()
                    self.get_loss()
                    if self.model.training:
                        self.backward()
                        self.step()
                        self.zero_grad()
    
    def fit(self, n_epochs):
        self.n_epochs = n_epochs
        self.epochs = range(n_epochs)
        self.opt = self.opt_func(self.model.parameters(), self.lr)
        with self.callback_ctx('fit'):
            for self.epoch in self.epochs:
                self.one_epoch(True)
                with torch.no_grad():
                    self.one_epoch(False)

    def __getattr__(self, name):
        if name in ('predict','get_loss','backward','step','zero_grad'):
            return partial(self.callback, name)
        raise AttributeError(name)

    def callback(self, method_nm):
        run_cbs(self.cbs, method_nm)
```

```{code-cell} ipython3
#|export
class TrainCB(Callback):
    def predict(self):
        self.learn.preds = self.learn.model(self.learn.batch[0])
    def get_loss(self):
        self.learn.loss = self.learn.loss_func(self.learn.preds, self.learn.batch[1])
    def backward(self):
        self.learn.loss.backward()
    def step(self):
        self.learn.opt.step()
    def zero_grad(self):
        self.learn.opt.zero_grad()
```

```{code-cell} ipython3
#|export
class ProgressCB(Callback):
    order = MetricsCB.order+1
    def __init__(self, plot=False): self.plot = plot
    def before_fit(self):
        self.learn.epochs = self.mbar = master_bar(self.learn.epochs)
        if hasattr(self.learn, 'metrics'): self.learn.metrics._log = self._log
        self.losses = []
    def _log(self, d):
        self.mbar.write(str(d))
    def before_epoch(self):
        self.learn.dl = progress_bar(self.learn.dl, leave=False, parent=self.mbar)
    def after_batch(self):
        self.learn.dl.comment = f'{self.learn.loss:.3f}'
        if self.plot and hasattr(self.learn, 'metrics') and self.learn.model.training:
            self.losses.append(self.learn.loss.item())
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

That's really cool.

```{code-cell} ipython3
#|export
class MomentumLearner(Learner):
    def __init__(self, model, dls, loss_func, lr, cbs, opt_func=optim.SGD, mom=0.85):
        self.mom = mom
        super().__init__(model, dls, loss_func, lr, cbs, opt_func)

    def predict(self): self.preds = self.model(self.batch[0])
    def get_loss(self): self.loss = self.loss_func(self.preds, self.batch[1])
    def backward(self): self.loss.backward()
    def step(self): self.opt.step()
    def zero_grad(self):
        with torch.no_grad():
            for p in self.model.parameters(): p.grad *= self.mom
```

```{code-cell} ipython3
# NB: No TrainCB
metrics = MetricsCB(accuracy=MulticlassAccuracy())
cbs = [DeviceCB(), metrics, ProgressCB(plot=True)]
learn = MomentumLearner(get_model(), dls, F.cross_entropy, lr=0.2, cbs=cbs)
learn.fit(1)
```

```{code-cell} ipython3
class LRFinderCB(Callback):
    def __init__(self, lr_mult=1.3):
        fc.store_attr()
    
    def before_fit(self):
        self.lrs,self.losses = [],[]
        self.min = math.inf

    def after_batch(self):
        if not self.learn.model.training:
            raise CancelEpochException()
        self.lrs.append(self.learn.opt.param_groups[0]['lr'])
        loss = to_cpu(self.learn.loss)
        self.losses.append(loss)
        if loss < self.min: self.min = loss
        if loss > self.min*3: raise CancelFitException()
        for g in self.learn.opt.param_groups:
            g['lr'] *= self.lr_mult
```

```{code-cell} ipython3
lrfind = LRFinderCB()
cbs = [DeviceCB(), ProgressCB(), lrfind]
learn = MomentumLearner(get_model(), dls, F.cross_entropy, lr=1e-4, cbs=cbs)
learn.fit(1)
```

```{code-cell} ipython3
plt.plot(lrfind.lrs, lrfind.losses)
plt.xscale('log')
```

The take away here is that we are defining callbacks and not modifying 
any of the actual training code. So we get all this extra magic/functionality by just creating and adding callbacks.

```{code-cell} ipython3
#|export
from torch.optim.lr_scheduler import ExponentialLR
```

```{code-cell} ipython3
#|export
class LRFinderCB(Callback):
    def __init__(self, gamma=1.3, max_mult=3): fc.store_attr()
    
    def before_fit(self):
        self.sched = ExponentialLR(self.learn.opt, self.gamma)
        self.lrs,self.losses = [],[]
        self.min = math.inf

    def after_batch(self):
        if not self.learn.model.training: raise CancelEpochException()
        self.lrs.append(self.learn.opt.param_groups[0]['lr'])
        loss = to_cpu(self.learn.loss)
        self.losses.append(loss)
        if loss < self.min: self.min = loss
        if loss > self.min*self.max_mult: raise CancelFitException()
        self.sched.step()

    def after_fit(self):
        plt.plot(self.lrs, self.losses)
        plt.xscale('log')
```

```{code-cell} ipython3
lrfind = LRFinderCB()
cbs = [DeviceCB(), ProgressCB(), lrfind]
learn = MomentumLearner(get_model(), dls, F.cross_entropy, lr=1e-4, cbs=cbs)
learn.fit(1)
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

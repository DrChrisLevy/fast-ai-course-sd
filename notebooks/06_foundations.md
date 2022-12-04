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

# Decorators
https://www.youtube.com/watch?v=FsAPt_9Bf3U

```{code-cell} ipython3
def decorator_function(original_function):
    def wrapper_function():
        print(f'Starting {original_function.__name__}')
        original_function()
        print(f'Done {original_function.__name__}')
    return wrapper_function
```

```{code-cell} ipython3
def display():
    print('I ran')
```

```{code-cell} ipython3
display()
```

```{code-cell} ipython3
modified_display = decorator_function(display)
```

```{code-cell} ipython3
modified_display()
```

So the decorator function is used to modify wrap some other logic around the original function without changing its functionality. 

Let's look at another example.

```{code-cell} ipython3
import time

def slow_add(*args, **kwargs):
    wait_time = kwargs.get('wait', 2)
    print(f'adding up {args} in {wait_time} seconds')
    time.sleep(wait_time)
    return sum(args)
    
```

```{code-cell} ipython3
slow_add(1,2,3,4)
```

```{code-cell} ipython3
slow_add(1,-1, wait=5)
```

```{code-cell} ipython3
def decorator_function(original_function):
    def wrapper_function(*args, **kwargs):
        print('starting slow add')
        original_function(*args, **kwargs)
        print('finishing slow add')
    return wrapper_function
    
```

```{code-cell} ipython3
modified_slow_add = decorator_function(slow_add)
```

```{code-cell} ipython3
modified_slow_add(1,2,wait=2)
```

That is what a decorator function is.
Put simply: decorators wrap a function, modifying its behavior.


We can use the `@` syntax
instead of the `modified_slow_add = decorator_function(slow_add)` notation.

```{code-cell} ipython3
@decorator_function
def slow_add(*args, **kwargs):
    wait_time = kwargs.get('wait', 2)
    print(f'adding up {args} in {wait_time} seconds')
    time.sleep(wait_time)
    return sum(args)
```

Now whenever we call `slow_add` it will do under the hood `decorator_function(slow_add)()`.

```{code-cell} ipython3
slow_add(1,2,wait=1.4)
```

Sometime people create decorators with classes.

Suppose we wanted some simple retry logic for this flaky function.

```{code-cell} ipython3
import random
def flakey_add(*args, **kwargs):
    should_fail = random.random() > 1 - kwargs.get('fail', 0.5)
    if should_fail:
        raise Exception('OOPS. FAILED')
    else:
        return sum(args)
```

```{code-cell} ipython3
for i in range(5):
    print(i)
    flakey_add(1,2)
```

```{code-cell} ipython3
class RetryLogic:
    def __init__(self, original_function):
        self.original_function = original_function
        
    def __call__(self, *args,**kwargs):
        i = 0
        while True:
            try:
                res = self.original_function(*args, **kwargs)
                print(f' attempt {i} was a success')
                return res
            except Exception:
                print(f' attempt {i} was a fail')
                i+=1
                continue
```

```{code-cell} ipython3
RetryLogic(flakey_add)(1,2,3,fail=0.8)
```

```{code-cell} ipython3
@RetryLogic
def flakey_add(*args, **kwargs):
    should_fail = random.random() > 1 - kwargs.get('fail', 0.5)
    if should_fail:
        raise Exception('OOPS. FAILED')
    else:
        return sum(args)
```

```{code-cell} ipython3
flakey_add(1,2,3,fail=0.8)
```

One more example with two decorator functions.

```{code-cell} ipython3
def log_decorator(original_function):
    def wrapper(*args, **kwargs):
        print(f'executing {original_function.__name__}')
        return original_function(*args, **kwargs)
    return wrapper

def timeit_decorator(original_function):
    def wrapper(*args, **kwargs):
        ct = time.time()
        res = original_function(*args, **kwargs)
        print(f'{original_function.__name__} ran in {time.time() - ct} seconds')
        return res
    return wrapper
```

```{code-cell} ipython3
def add_em_up(*args, message='HELLO WORLD!'):
    print(message)
    return sum(args)
```

```{code-cell} ipython3
add_em_up(4,5,6,message='hey there')
```

```{code-cell} ipython3
@log_decorator
def add_em_up(*args, message='HELLO WORLD!'):
    print(message)
    return sum(args)
```

```{code-cell} ipython3
add_em_up(4,5,6,message='hey there')
```

```{code-cell} ipython3
@timeit_decorator
def add_em_up(*args, message='HELLO WORLD!'):
    print(message)
    return sum(args)
```

```{code-cell} ipython3
add_em_up(4,5,6,message='hey there')
```

```{code-cell} ipython3
@log_decorator
@timeit_decorator
def add_em_up(*args, message='HELLO WORLD!'):
    print(message)
    return sum(args)
```

```{code-cell} ipython3
add_em_up(4,5,6,message='hey there')
```

Note that the `__name__` got mangled here. We can fix that
by using `from functools import wraps`

```{code-cell} ipython3
from functools import wraps

def log_decorator(original_function):
    @wraps(original_function)
    def wrapper(*args, **kwargs):
        print(f'executing {original_function.__name__}')
        return original_function(*args, **kwargs)
    return wrapper

def timeit_decorator(original_function):
    @wraps(original_function)
    def wrapper(*args, **kwargs):
        ct = time.time()
        res = original_function(*args, **kwargs)
        print(f'{original_function.__name__} ran in {time.time() - ct} seconds')
        return res
    return wrapper
```

```{code-cell} ipython3
@log_decorator
@timeit_decorator
def add_em_up(*args, message='HELLO WORLD!'):
    print(message)
    return sum(args)
```

```{code-cell} ipython3
add_em_up(4,5,6,message='hey there')
```

Some practical examples of decorators
- logging
- timing
- retries
and lots of other things.

+++

# Partial Functions
Starting with a function with some parameters, and then
set some of those parameters to constants and leave some of
the parameters as args in the new partial function.

```{code-cell} ipython3
from functools import partial

def add_abc(a,b,c):
    return a + b + c
```

```{code-cell} ipython3
f = partial(add_abc,4,8) # goes from left to right so now a=4 and c=8
```

```{code-cell} ipython3
f(0) # c=0 --> 4 + 8 + 0
```

```{code-cell} ipython3
f(-12) #--> 4 + 8 + -12
```

Another example

```{code-cell} ipython3
def myf(a,b,c):
    return 2 * a + b - c
```

```{code-cell} ipython3
f = partial(myf, b=0,c=0)
```

```{code-cell} ipython3
f(2)
```

# Callbacks

```{code-cell} ipython3
def show_calc(epochs, cb=None):
    res = 0
    for epoch in range(epochs):
        time.sleep(1)
        res += 1
        if cb:
            cb(epoch)
    return res
```

```{code-cell} ipython3
show_calc(4)
```

```{code-cell} ipython3
def show_progress(epoch):
    print(f'Finished epoch {epoch}')
show_calc(4, show_progress) # show_progress function is the call back
```

```{code-cell} ipython3
show_calc(4, lambda epoch: print(f'Finished epoch {epoch}'))
```

## Callbacks as Callable Classes

```{code-cell} ipython3
class ProgressShowingCallback():
    def __init__(self, msg):
        self.msg = msg
        
    def __call__(self, epoch):
        print(f'{self.msg}. Finished epoch {epoch}')
```

```{code-cell} ipython3
cb = ProgressShowingCallback('Random Message')
```

```{code-cell} ipython3
show_calc(5, cb)
```

## Multiple Callbacks

```{code-cell} ipython3
def show_calc(epochs, cb=None):
    res = 0
    for epoch in range(epochs):
        if cb:
            cb.before_calc(epoch)
        time.sleep(1)
        res += 1
        if cb:
            cb.after_calc(epoch, res)
    return res
```

```{code-cell} ipython3
class PrintStepCallback:
    def __init__(self):
        pass
    
    def before_calc(self, epoch, **kwargs):
        print(f'starting epoch {epoch}')
        
    def after_calc(self, epoch, res, **kwargs):
        print(f'finishing epoch {epoch} and got {res}')
```

```{code-cell} ipython3
show_calc(5, PrintStepCallback())
```

## Modify Behavior

```{code-cell} ipython3
def show_calc(epochs, cb=None):
    res = 0
    for epoch in range(epochs):
        if cb and hasattr(cb, 'before_calc'):
            cb.before_calc(epoch)
        time.sleep(1)
        res += 1
        if cb and hasattr(cb, 'after_calc'):
            if cb.after_calc(epoch, res):
                print('early stopping')
                break
    return res

class PrintAfterCallback:
    def __init__(self):
        pass
    
    def before_calc(self, epoch, **kwargs):
        print(f'starting epoch {epoch}')
        
    def after_calc(self, epoch, res, **kwargs):
        print(f'finishing epoch {epoch} and got {res}')
        if res > 5:
            return True
```

```{code-cell} ipython3
show_calc(10, PrintAfterCallback())
```

Call backs can be functions, callable's classes, etc.
There is a lot of flexibility through Python.

```{code-cell} ipython3
class SlowCalc:
    def __init__(self, cb=None):
        self.res = 0
        self.cb = cb
    
    def callback(self, cb_name, *args, **kwargs):
        if self.cb is None:
            return
        cb = getattr(self.cb, cb_name, None)
        if cb:
            return cb(*args, **kwargs)
        
    def calc(self, epochs):
        for epoch in range(epochs):
            self.callback('before_calc', self, epoch)
            time.sleep(1)
            self.res+=1
            if self.callback('after_calc', self, epoch):
                print('EARLY STOP')
                break
```

```{code-cell} ipython3
class ModifyingCallback:
    def __init__(self):
        pass
    
    def before_calc(self, calc, epoch):
        print(f'doing epoch {epoch} and val is {calc.res}')
        
    def after_calc(self, calc, epoch):
        print(f'finish epoch {epoch} and val is {calc.res}')
        if calc.res <20:
            calc.res = calc.res**2
        if calc.res > 25:
            return True
```

```{code-cell} ipython3
SlowCalc(ModifyingCallback()).calc(10)
```

#  `__dunder__` thingies

+++

Special methods you should probably know about (see data model link above) are:

- `__getitem__`
- `__getattr__`
- `__setattr__`
- `__del__`
- `__init__`
- `__new__`
- `__enter__`
- `__exit__`
- `__len__`
- `__repr__`
- `__str__`

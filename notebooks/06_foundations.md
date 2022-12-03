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

```{code-cell} ipython3

```

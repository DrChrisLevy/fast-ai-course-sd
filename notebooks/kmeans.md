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
import torch
```

We need some functionality to find the distance between a set of vectors
in one tensor with the set of vectors in another tensor.

```{code-cell} ipython3
Xb =  torch.tensor([[1.,2.,3.],[4.,5.,6.], [7.,8.,9.]])
Xq  = torch.tensor([[4.,5.,6.], [10., 11. , 12.], [13., 14. , 15.], [16., 17. , 18.]])
```

```{code-cell} ipython3
Xq, Xb
```

```{code-cell} ipython3
# simple loop for unit test
def ground_truth(Xq, Xb):
    distances = torch.zeros((Xq.shape[0],Xb.shape[0]))
    for i in range(len(Xq)):
        for j in range(len(Xb)):
            x = Xq[i]
            y = Xb[j]
            distances[i,j] = (x-y).square().sum()
    return distances
```

```{code-cell} ipython3
ground_truth(Xq, Xb)
```

```{code-cell} ipython3
ground_truth(Xb, Xq)
```

```{code-cell} ipython3
print(Xq.shape)
print(Xb.shape)
```

Okay so the 4 and 3 don't match so broadcasting won't work.
So lets make that "4" a "1" in the shape of `Xq`

```{code-cell} ipython3
print(Xq[:, None, :].shape)
print(Xb.shape)
```

Now we need to add a "1" at the start of the `Xb` shape

```{code-cell} ipython3
print(Xq[:, None, :].shape) # being explicit here but can be written as Xq[:, None]
print(Xb[None,:, :].shape)  # being explicit here but can be written as Xb[None]
```

```{code-cell} ipython3
assert torch.equal(Xq[:, None, :] , Xq[:,None])
assert torch.equal(Xb[None,:, :] , Xb[None])
```

So now the above two tensors, `Xq[:,None]` and `Xb[None])` would allow for broadcast rules to be applied.

```{code-cell} ipython3
print(Xq[:,None].shape)
print(Xb[None].shape)
```

```{code-cell} ipython3
(Xq[:, None] - Xb[None])
```

```{code-cell} ipython3
(Xq[:, None] - Xb[None]).shape
```

```{code-cell} ipython3
(Xq[:, None] - Xb[None]).square()
```

```{code-cell} ipython3
(Xq[:, None] - Xb[None]).square().sum(2)
```

What is this magic??

```{code-cell} ipython3
assert torch.equal(ground_truth(Xq, Xb), (Xq[:, None] - Xb[None]).square().sum(2))
```

```{code-cell} ipython3
assert torch.equal(ground_truth(Xb, Xq), (Xb[:, None] - Xq[None]).square().sum(2))
```

```{code-cell} ipython3
def square_distance(Xb, Xq):
    return (Xb[:, None] - Xq[None]).square().sum(2)
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

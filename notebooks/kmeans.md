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

# Pairwise Distances

+++

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

# Create Some Fake Cluster Data

```{code-cell} ipython3
n_clusters=6
n_samples =250
torch.manual_seed(42)

centroids_truth = torch.rand(n_clusters, 2)*70-35

from matplotlib import pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import tensor

def sample(m): return MultivariateNormal(m, torch.diag(tensor([5.,5.]))).sample((n_samples,))

slices = [sample(c) for c in centroids_truth]
data = torch.cat(slices)
data.shape


def plot_data(centroids_truth, data, n_samples, ax=None):
    if ax is None: _,ax = plt.subplots()
    for i, centroid in enumerate(centroids_truth):
        samples = data[i*n_samples:(i+1)*n_samples]
        ax.scatter(samples[:,0], samples[:,1], s=1)
        ax.plot(*centroid, markersize=10, marker="x", color='k', mew=5)
        ax.plot(*centroid, markersize=5, marker="x", color='m', mew=2)

plot_data(centroids_truth, data, n_samples)
```

## Initialization

For kmeans to work well we need a good random initialization 
for the centroids. 

One option is to choose $k$ random points from the dataset.
Run this cell multiple times to see that this method
is quite volatile. Often the initially random selected centroids
will be from the same cluster.

```{code-cell} ipython3
k = 6
centroids = data[random.sample(range(len(data)),6)] # this random initialization method is not very good!
plot_data(centroids, data, n_samples)
```

[kmeans++](https://en.wikipedia.org/wiki/K-means%2B%2B) is a better initialization method.

- Choose one centroids uniformly at random among the data
For each data point x not chosen yet, compute D(x), the distance between x and the nearest center that has already been chosen.
Choose one new data point at random as a new center, using a weighted probability distribution where a point x is chosen with probability proportional to D(x)2.
Repeat Steps 2 and 3 until k centers have been chosen.
Now that the initial centers have been chosen, proceed using standard k-means clustering.

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3
dists = square_distance(data, centroids)
dists
```

```{code-cell} ipython3
assigned_clusters = torch.argmin(dists, dim=1)
print(assigned_clusters.shape)
assigned_clusters
```

```{code-cell} ipython3
for j in range(k): # TODO how to do without a for loop
    centroids[j] = data[assigned_clusters == j].mean(dim=0)
centroids
```

```{code-cell} ipython3
plot_data(centroids, data, n_samples)
```

```{code-cell} ipython3
def kmeans(data, k, iterations=5):
    # initialize random centroids
    centroids = data[random.sample(range(0, len(data)), k), :]
    
    for i in range(iterations):
        dists = square_distance(data, centroids)
        assigned_clusters = torch.argmin(dists, dim=1)
        
        for j in range(k): # TODO how to do without a for loop
            centroids[j] = data[assigned_clusters == j].mean(dim=0)
    return centroids, assigned_clusters
```

```{code-cell} ipython3
centroids, assigned_clusters = kmeans(data, 6, 5) 
plot_data(centroids, data, n_samples)
```

```{code-cell} ipython3

```

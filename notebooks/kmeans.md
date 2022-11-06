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
import random
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

What is this magic?? Broadcasting!

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
n_samples =2500
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

# Initialization of Centroids

## Random Initialization

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

## kmeans++ (better random initialization)

+++

[kmeans++](https://en.wikipedia.org/wiki/K-means%2B%2B) is a better initialization method.
The idea is to spread out the initial centroids as much as possible.

- pick a centroid at random from the dataset.
- compute the distance between each data point and the centroid.
- sample another point from the dataset in such a way that a point that is further away from the first centroid has a higher probability of being selected. Now there are two centroids selected.
- Now compute the distance between each point and the nearest centroid. Similarly, pick another point randomly from the dataset using weighted random sampling (based on the distance from the data point to the nearest centroid).
- Continue until you have $k$ centroids.

```{code-cell} ipython3
centroids = data[random.sample(range(len(data)),1)] # pick first point at random
```

```{code-cell} ipython3
plot_data(centroids, data, n_samples)
```

```{code-cell} ipython3
dists_to_centroids = square_distance(data, centroids)
dists_to_centroids = dists_to_centroids.min(dim=1)[0] # get the distance to nearest centroid
weights = (dists_to_centroids / dists_to_centroids.max().square())
weights
```

```{code-cell} ipython3
from torch.utils.data import WeightedRandomSampler
```

```{code-cell} ipython3
centroid_idx = list(WeightedRandomSampler(weights, 1, replacement=False))[0]
centroid_idx
```

```{code-cell} ipython3
centroids = torch.vstack([centroids, data[centroid_idx]])
```

```{code-cell} ipython3
plot_data(centroids, data, n_samples)
```

```{code-cell} ipython3
dists_to_centroids = square_distance(data, centroids)
dists_to_centroids = dists_to_centroids.min(dim=1)[0] # get the distance to nearest
weights = (dists_to_centroids / dists_to_centroids.max().square())
centroid_idx = list(WeightedRandomSampler(weights, 1, replacement=False))[0]
centroids = torch.vstack([centroids, data[centroid_idx]])
plot_data(centroids, data, n_samples)
```

And so on...

Lets wrap the logic for initialization in a single function:

```{code-cell} ipython3
def initialize_centroids(data, k):
    centroids = data[random.sample(range(len(data)),1)] # pick first point at random
    
    for i in range(k-1): # pick remaining centroids
        dists_to_centroids = square_distance(data, centroids)
        dists_to_centroids = dists_to_centroids.min(dim=1)[0] # get the distance to nearest
        weights = (dists_to_centroids / dists_to_centroids.max().square())
        centroid_idx = list(WeightedRandomSampler(weights, 1, replacement=False))[0]
        centroids = torch.vstack([centroids, data[centroid_idx]])
    
    return centroids
    
```

```{code-cell} ipython3
centroids = initialize_centroids(data, k)
print(centroids)
plot_data(centroids, data, n_samples)
```

## Iterations for Updating Centroids

Now we have the initial centroids selected.
Next we proceed with the kmeans implementation:

- compute distances between data points and centroids
- map each data point to nearest centroid
- update centroids based on the data points in each cluster
- repeat until it stabilizes 

Lets go through one iteration here to show how it works.

```{code-cell} ipython3
dists = square_distance(data, centroids)
dists # torch.Size([N, k]) holds the distance between each data point and each centroid
```

```{code-cell} ipython3
assigned_clusters = torch.argmin(dists, dim=1) # pick the centroid with the smallest distance
print(assigned_clusters.shape)
assigned_clusters
```

```{code-cell} ipython3
# update the centroids by taking the centroid of each cluster
for j in range(k): # TODO how to do without a for loop
    centroids[j] = data[assigned_clusters == j].mean(dim=0)
centroids
```

```{code-cell} ipython3
plot_data(centroids, data, n_samples)
```

Okay, lets wrap it all up in one function.

```{code-cell} ipython3
def kmeans(data, k, iterations=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # initialize centroids
    centroids = initialize_centroids(data, k).to(device)
    data = data.to(device)
    
    # update centroids though iteration
    for i in range(iterations):
        dists = square_distance(data, centroids)
        assigned_clusters = torch.argmin(dists, dim=1)
        
        for j in range(k): # TODO how to do without a for loop
            centroids[j] = data[assigned_clusters == j].mean(dim=0)
    return centroids, assigned_clusters
```

```{code-cell} ipython3
centroids, assigned_clusters = kmeans(data, 6, 5) 
plot_data(centroids.cpu(), data.cpu(), n_samples)
```

```{code-cell} ipython3
%timeit centroids, assigned_clusters = kmeans(data, 6, 5)
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

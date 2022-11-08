import torch
from torch.utils.data import WeightedRandomSampler


def square_distance(Xb, Xq):
    return (Xb[:, None] - Xq[None]).square().sum(2)


def initialize_centroids(data, k):
    centroids = data[torch.randint(0, len(data), (1,))]  # pick first point at random

    for i in range(k - 1):  # pick remaining centroids
        dists_to_centroids = square_distance(data, centroids)
        dists_to_centroids = dists_to_centroids.min(dim=1)[0]  # get the distance to nearest
        weights = dists_to_centroids / dists_to_centroids.max().square()
        centroid_idx = list(WeightedRandomSampler(weights, 1, replacement=False))[0]
        centroids = torch.vstack([centroids, data[centroid_idx]])

    return centroids


def kmeans(data, k, iterations=10):
    # initialize centroids
    centroids = initialize_centroids(data, k)
    data = data

    # update centroids though iteration
    for i in range(iterations):
        dists = square_distance(data, centroids)
        assigned_clusters = torch.argmin(dists, dim=1)

        for j in range(k):  # TODO how to do without a for loop
            centroids[j] = data[assigned_clusters == j].mean(dim=0)
    return centroids, assigned_clusters

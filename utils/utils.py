import torch
from torch.utils.data import WeightedRandomSampler
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


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


class LRFinderCallback(keras.callbacks.Callback):
    def __init__(self, lr_factor=1.1, max_lr=1e-1):
        super().__init__()
        self.factor = lr_factor
        self.lrs = []
        self.losses = []
        self.max_lr = max_lr

    def get_lr(self):
        return float(keras.backend.get_value(self.model.optimizer.learning_rate))

    def set_lr(self, lr):
        keras.backend.set_value(self.model.optimizer.lr, lr)

    def on_train_batch_end(self, batch, logs=None):
        lr = self.get_lr()
        self.lrs.append(lr)
        self.losses.append(logs["loss"])

        if lr > self.max_lr:
            self.model.stop_training = True
            plt.plot(np.log10(self.lrs), self.losses)

        self.set_lr(lr * self.factor)
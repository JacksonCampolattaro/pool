import torch


def naive_pool(features, neighbors):
    edge_features = features[neighbors, :]
    return edge_features.amax(dim=1)

import torch


def naive_pool(features: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
    edge_features = features[neighbors, :]
    return edge_features.amax(dim=1)
    # return edge_features.max(dim=1)[0]

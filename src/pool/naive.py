import torch


def max_pool(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    edge_features = x[index, :]
    return edge_features.max(dim=1)[0] # TODO: This is different from amax()!

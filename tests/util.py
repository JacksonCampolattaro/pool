from typing import Callable

import pytest
import torch
import torch_geometric
from torch_geometric import EdgeIndex
from torch_geometric.nn import SimpleConv
from torch_sparse import SparseTensor

from pool.cuda import cuda_pool
from pool.naive import naive_pool

VALUE_DTYPES = [
    pytest.param(torch.half, id='half'),
    pytest.param(torch.float16, id='float16'),
    pytest.param(torch.bfloat16, id='bfloat16'),
    pytest.param(torch.float32, id='float32'),
    pytest.param(torch.float64, id='float64'),
]


def generate_input_data(
        m: int = 8192, n: int = 4096, k: int = 24, c: int = 128,
        device: str = 'cpu', dtype: torch.dtype = None, requires_grad: bool = False
):
    dtype = dtype if dtype else (torch.float32 if device == 'cpu' else torch.float16)
    features = torch.randn([m, c]).to(device=device, dtype=dtype)
    if requires_grad:
        features.requires_grad_()
    neighbors = torch.randint(m, size=[n, k]).to(device=device)
    return features, neighbors


def to_edges(features: torch.Tensor, neighbors: torch.Tensor):
    return EdgeIndex(torch.stack([
        neighbors.flatten(),
        torch.arange(neighbors.size(0)).repeat_interleave(neighbors.size(1)).to(device=neighbors.device),
    ]), sparse_size=(features.size(0), neighbors.size(0))).validate()


def to_sparse_edges(features: torch.Tensor, neighbors: torch.Tensor):
    return SparseTensor(
        row=torch.arange(
            neighbors.size(0),
            device=neighbors.device, dtype=neighbors.dtype
        ).unsqueeze(-1).expand([-1, neighbors.size(1)]).flatten(),
        col=neighbors.sort(dim=-1)[0].flatten(),
        sparse_sizes=(neighbors.size(0), features.size(0)),
        is_sorted=True,
        trust_data=True,
    )


pyg_conv = SimpleConv(aggr='max')


def pyg_pool(features: torch.Tensor, edges: torch_geometric.EdgeIndex | SparseTensor):
    return pyg_conv((features, None), edges)


sparse_pool = pyg_pool


def tosparse_pool(features, neighbors):
    sparse_edges = to_sparse_edges(features, neighbors)
    return pyg_conv((features, None), sparse_edges)


def edges_for_pool_function(features, neighbors, function: Callable):
    if function == cuda_pool and not features.is_cuda:
        pytest.skip("Cannot test cuda kernel on other devices")
    if features.dtype == torch.bfloat16 and function in (sparse_pool, tosparse_pool):
        pytest.skip("torch_sparse doesn't seem to support bf16 values")

    if function == pyg_pool:
        return to_edges(features, neighbors)
    if function == sparse_pool:
        return to_sparse_edges(features, neighbors)
    else:
        return neighbors


POOL_FUNCTIONS = [
    pytest.param(naive_pool, id='naive_pool'),
    pytest.param(pyg_pool, id='pyg_pool'),
    pytest.param(sparse_pool, id='sparse_pool'),
    pytest.param(tosparse_pool, id='tosparse_pool'),
    pytest.param(cuda_pool, id='cuda_pool'),
]

from typing import Callable

import pytest
import torch
import torch_geometric
from torch_geometric import EdgeIndex
from torch_geometric.nn import SimpleConv
from torch_sparse import SparseTensor

from pool.cuda import max_pool as cuda_pool
from pool.naive import max_pool as naive_pool

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
    x = torch.randn([m, c], device=device, dtype=dtype, requires_grad=requires_grad)
    index = torch.randint(m, size=[n, k]).to(device=device)
    return x, index


def to_edges(x: torch.Tensor, index: torch.Tensor):
    return EdgeIndex(torch.stack([
        index.flatten(),
        torch.arange(index.size(0)).repeat_interleave(index.size(1)).to(device=index.device),
    ]), sparse_size=(x.size(0), index.size(0))).validate()


def to_sparse_edges(x: torch.Tensor, index: torch.Tensor):
    return SparseTensor(
        row=torch.arange(
            index.size(0),
            device=index.device, dtype=index.dtype
        ).unsqueeze(-1).expand([-1, index.size(1)]).flatten(),
        col=index.sort(dim=-1)[0].flatten(),
        sparse_sizes=(index.size(0), x.size(0)),
        is_sorted=True,
        trust_data=True,
    )


pyg_conv = SimpleConv(aggr='max')


def pyg_pool(x: torch.Tensor, edges: torch_geometric.EdgeIndex | SparseTensor):
    return pyg_conv((x, None), edges)


sparse_pool = pyg_pool


def tosparse_pool(x, index):
    sparse_edges = to_sparse_edges(x, index)
    return pyg_conv((x, None), sparse_edges)


def edges_for_pool_function(x, index, function: Callable):
    if function == cuda_pool and not x.is_cuda:
        pytest.skip("Cannot test cuda kernel on other devices")
    if x.dtype == torch.bfloat16 and function in (sparse_pool, tosparse_pool):
        pytest.skip("torch_sparse doesn't seem to support bf16 values")

    if function == pyg_pool:
        return to_edges(x, index)
    if function == sparse_pool:
        return to_sparse_edges(x, index)
    else:
        return index


POOL_FUNCTIONS = [
    pytest.param(naive_pool, id='naive_pool'),
    # pytest.param(pyg_pool, id='pyg_pool'),
    # pytest.param(sparse_pool, id='sparse_pool'),
    # pytest.param(tosparse_pool, id='tosparse_pool'),
    pytest.param(cuda_pool, id='cuda_pool'),
    # pytest.param(torch.compile(cuda_pool), id='compiled_cuda_pool'),
]

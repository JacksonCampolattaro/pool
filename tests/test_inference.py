from typing import Callable

import pytest
import torch
import torch_geometric
from torch_geometric import EdgeIndex
from torch_geometric.nn import SimpleConv
from torch_geometric.testing.decorators import withDevice as parameterizeByDevice
from torch_sparse import SparseTensor

from pool.naive import naive_pool
from pool.cuda import cuda_pool

TEST_CONFIGS_M_N_K_C = [
    (1024, 128, 24, 32),
    (128, 1024, 24, 32),
    (1024 * 4, 1024 * 4, 24, 32),
    # (1024 * 64, 1024 * 64, 24, 256),
    # (1024 * 256, 1024 * 256, 24, 64),
]

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


class TestInference:

    @parameterizeByDevice
    @pytest.mark.parametrize('config', TEST_CONFIGS_M_N_K_C)
    @pytest.mark.parametrize('pool_function', POOL_FUNCTIONS)
    @pytest.mark.parametrize('requires_grad', [True, False])
    @pytest.mark.parametrize('dtype', VALUE_DTYPES)
    def test_inference_correctness(
            self,
            device: str,
            config: tuple,
            pool_function: Callable,
            requires_grad: bool,
            dtype: torch.dtype
    ):
        m, n, k, c = config
        features, neighbors = generate_input_data(*config, device=device, dtype=dtype, requires_grad=requires_grad)
        edges = edges_for_pool_function(features, neighbors, pool_function)

        pooled = pool_function(features, edges)

        assert pooled.shape == (n, features.size(-1))
        first_neighbor = features[neighbors[:, 0], :]
        assert (pooled >= first_neighbor).all()
        assert torch.isclose(pooled, naive_pool(features, neighbors)).all()

    @parameterizeByDevice
    @pytest.mark.parametrize('config', TEST_CONFIGS_M_N_K_C)
    @pytest.mark.parametrize('pool_function', POOL_FUNCTIONS)
    def test_inference_speed(self, benchmark, device: str, config: tuple, pool_function: Callable):
        features, neighbors = generate_input_data(*config, device=device)
        edges = edges_for_pool_function(features, neighbors, pool_function)

        benchmark(pool_function, features, edges)

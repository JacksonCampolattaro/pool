from typing import Callable

import torch
import pytest

from torch_geometric.nn.conv import SimpleConv
from torch_sparse import SparseTensor

from pool.naive import naive_pool
from pool.cuda import cuda_pool

from .test_inference import parameterizeByDevice, TEST_CONFIGS_M_N_K_C
from .util import VALUE_DTYPES, generate_input_data, edges_for_pool_function, POOL_FUNCTIONS

SMALL_TEST_CONFIGS_M_N_K_C = [
    (32, 32, 8, 16),
    (32, 64, 8, 16),
    (64, 32, 8, 16),
]

class TestBackprop:

    @parameterizeByDevice
    @pytest.mark.parametrize('config', SMALL_TEST_CONFIGS_M_N_K_C)
    @pytest.mark.parametrize('pool_function', POOL_FUNCTIONS)
    @pytest.mark.parametrize('dtype', VALUE_DTYPES)
    def test_backprop_correctness(
            self,
            device: str,
            config: tuple,
            pool_function: Callable,
            dtype: torch.dtype
    ):
        features, neighbors = generate_input_data(*config, device=device, dtype=dtype, requires_grad=True)
        features_ref = features.clone().detach().requires_grad_(True)
        edges = edges_for_pool_function(features, neighbors, pool_function)

        out = pool_function(features, edges)
        out_ref = naive_pool(features_ref, neighbors)

        # todo: way too lax
        torch.autograd.gradcheck(pool_function, (features, edges), eps=1, atol=1, nondet_tol=1)
        #
        # out.sum().backward()
        # out_ref.sum().backward()
        #
        # diff = features.grad - features_ref.grad
        # close = torch.isclose(features.grad, features_ref.grad, atol=1e-4)
        # # print(diff[~close], (~close).nonzero().size(0))
        # print((~close).float().mean())
        # # print(features.grad - features_ref.grad)
        # # print(torch.isclose(features.grad, features_ref.grad, atol=1e-4))
        # assert (~close).float().mean() < 0.001
        # assert torch.isclose(features.grad, features_ref.grad, atol=1e-4).all()

    @parameterizeByDevice
    @pytest.mark.parametrize('config', TEST_CONFIGS_M_N_K_C)
    @pytest.mark.parametrize('pool_function', POOL_FUNCTIONS)
    def test_roundtrip_speed(self, benchmark, device: str, config: tuple, pool_function: Callable):
        features, neighbors = generate_input_data(*config, device=device, requires_grad=True)
        edges = edges_for_pool_function(features, neighbors, pool_function)

        def fwd_and_bwd():
            pool_function(features, edges).sum().backward()

        benchmark(fwd_and_bwd)

    # @parameterizeByDevice
    # @pytest.mark.parametrize('config', TEST_CONFIGS_M_N_K_C)
    # def test_naive_pool(self, benchmark, device: str, config: tuple):
    #     features, neighbors = generate_input_data(*config, device=device, requires_grad=True)
    #
    #     def fwd_and_bwd():
    #         naive_pool(features, neighbors).sum().backward()
    #
    #     benchmark(fwd_and_bwd)
    #
    # @parameterizeByDevice
    # @pytest.mark.parametrize('config', TEST_CONFIGS_M_N_K_C)
    # def test_pyg_pool(self, benchmark, device, config: tuple):
    #     features, neighbors = generate_input_data(*config, device=device, requires_grad=True)
    #     edges = to_edges(features, neighbors)
    #
    #     def fwd_and_bwd():
    #         pyg_pool(features, edges).sum().backward()
    #
    #     benchmark(fwd_and_bwd)
    #
    # @parameterizeByDevice
    # @pytest.mark.parametrize('config', TEST_CONFIGS_M_N_K_C)
    # def test_sparse_pool(self, benchmark, device, config: tuple):
    #     features, neighbors = generate_input_data(*config, device=device, requires_grad=True)
    #     sparse_edges = to_sparse_edges(features, neighbors)
    #
    #     def fwd_and_bwd():
    #         pyg_pool(features, sparse_edges).sum().backward()
    #
    #     benchmark(fwd_and_bwd)
    #
    # @parameterizeByDevice
    # @pytest.mark.parametrize('config', TEST_CONFIGS_M_N_K_C)
    # def test_tosparse_pool(self, benchmark, device, config: tuple):
    #     features, neighbors = generate_input_data(*config, device=device, requires_grad=True)
    #
    #     def fwd_and_bwd():
    #         tosparse_pool(features, neighbors).sum().backward()
    #
    #     benchmark(fwd_and_bwd)
    #
    # @pytest.mark.parametrize('config', TEST_CONFIGS_M_N_K_C)
    # def test_cuda_pool(self, benchmark, config: tuple):
    #     features, neighbors = generate_input_data(*config, device='cuda', requires_grad=True)
    #
    #     def fwd_and_bwd():
    #         cuda_pool(features, neighbors).sum().backward()
    #
    #     benchmark(fwd_and_bwd)

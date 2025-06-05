import torch
import pytest

from torch_geometric.nn.conv import SimpleConv
from torch_sparse import SparseTensor

from pool.naive import naive_pool
from pool.cuda import cuda_pool

from .test_performance import parameterizeByDevice, generate_input_data, to_sparse_edges, TEST_CONFIGS_M_N_K_C


class TestBackprop:
    @parameterizeByDevice
    @pytest.mark.parametrize('config', TEST_CONFIGS_M_N_K_C)
    def test_naive_pool(self, benchmark, device: str, config: tuple):
        features, neighbors, _ = generate_input_data(*config, device=device)

        def fwd_and_bwd():
            naive_pool(features, neighbors).sum().backward()

        benchmark(fwd_and_bwd)


    @parameterizeByDevice
    @pytest.mark.parametrize('config', TEST_CONFIGS_M_N_K_C)
    def test_pyg_pool(self, benchmark, device, config: tuple):
        features, neighbors, edges = generate_input_data(*config, device=device)
        conv = SimpleConv(aggr='max')

        def fwd_and_bwd():
            conv((features, None), edges, size=(features.size(0), neighbors.size(0))).sum().backward()

        benchmark(fwd_and_bwd)


    @parameterizeByDevice
    @pytest.mark.parametrize('config', TEST_CONFIGS_M_N_K_C)
    def test_sparse_pool(self, benchmark, device, config: tuple):
        features, neighbors, _ = generate_input_data(*config, device=device)
        sparse_edges = to_sparse_edges(features, neighbors)
        conv = SimpleConv(aggr='max')

        def fwd_and_bwd():
            conv((features, None), sparse_edges, size=(features.size(0), neighbors.size(0))).sum().backward()

        benchmark(fwd_and_bwd)


    @parameterizeByDevice
    @pytest.mark.parametrize('config', TEST_CONFIGS_M_N_K_C)
    def test_tosparse_pool(self, benchmark, device, config: tuple):
        features, neighbors, _ = generate_input_data(*config, device=device)
        conv = SimpleConv(aggr='max')

        def fwd_and_bwd():
            sparse_edges = to_sparse_edges(features, neighbors)
            conv((features, None), sparse_edges, size=(features.size(0), neighbors.size(0))).sum().backward()

        benchmark(fwd_and_bwd)


    @pytest.mark.parametrize('config', TEST_CONFIGS_M_N_K_C)
    def test_cuda_pool(self, benchmark, config: tuple):
        features, neighbors, _ = generate_input_data(*config, device='cuda')

        def fwd_and_bwd():
            cuda_pool(features, neighbors).sum().backward()

        benchmark(fwd_and_bwd)

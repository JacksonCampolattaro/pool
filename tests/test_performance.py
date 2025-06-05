import pytest
import torch
from torch_geometric import EdgeIndex
from torch_geometric.nn import SimpleConv
from torch_geometric.testing.decorators import withDevice as parameterizeByDevice
from torch_sparse import SparseTensor

from pool.naive import naive_pool

TEST_CONFIGS_M_N_K_C = [
    # (1024 * 4, 1024 * 4, 24, 32),
    # (1024 * 64, 1024 * 64, 24, 256),
    (1024 * 256, 1024 * 256, 24, 64),
]


def generate_input_data(m: int = 8192, n: int = 4096, k: int = 24, c: int = 128, device: str = 'cpu'):
    features = torch.rand([m, c]).to(device=device, dtype=torch.float if device == 'cpu' else torch.float16)
    features.requires_grad_()
    neighbors = torch.randint(m, size=[n, k]).to(device=device)
    edges = EdgeIndex(torch.stack([
        neighbors.flatten(),
        torch.arange(n).repeat_interleave(k).to(device=device),
    ]), sparse_size=(m, n)).validate()
    return features, neighbors, edges


def to_sparse_edges(features, neighbors):
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


class TestInference:

    @parameterizeByDevice
    @pytest.mark.parametrize('config', TEST_CONFIGS_M_N_K_C)
    def test_naive_pool(self, benchmark, device: str, config: tuple):
        m, n, k, c = config
        features, neighbors, _ = generate_input_data(*config, device=device)
        pooled = benchmark(naive_pool, features, neighbors)
        assert pooled.shape == (n, features.size(-1))
        first_neighbor = features[neighbors[:, 0], :]
        assert (pooled >= first_neighbor).all()

    @parameterizeByDevice
    @pytest.mark.parametrize('config', TEST_CONFIGS_M_N_K_C)
    def test_pyg_pool(self, benchmark, device, config: tuple):
        features, neighbors, edges = generate_input_data(*config, device=device)
        conv = SimpleConv(aggr='max')
        pooled = benchmark(conv, (features, None), edges, size=(features.size(0), neighbors.size(0)))
        assert (torch.abs(pooled - naive_pool(features, neighbors)) < 0.01).all()

    @parameterizeByDevice
    @pytest.mark.parametrize('config', TEST_CONFIGS_M_N_K_C)
    def test_sparse_pool(self, benchmark, device, config: tuple):
        features, neighbors, _ = generate_input_data(*config, device=device)
        sparse_edges = to_sparse_edges(features, neighbors)
        conv = SimpleConv(aggr='max')
        pooled = benchmark(conv, (features, None), sparse_edges, size=(features.size(0), neighbors.size(0)))
        assert (torch.abs(pooled - naive_pool(features, neighbors)) < 0.01).all()

    @parameterizeByDevice
    @pytest.mark.parametrize('config', TEST_CONFIGS_M_N_K_C)
    def test_tosparse_pool(self, benchmark, device, config: tuple):
        features, neighbors, _ = generate_input_data(*config, device=device)
        conv = SimpleConv(aggr='max')

        def tosparse_pool(features, neighbors):
            sparse_edges = to_sparse_edges(features, neighbors)
            return conv((features, None), sparse_edges, size=(features.size(0), neighbors.size(0)))

        pooled = benchmark(tosparse_pool, features, neighbors)
        assert (torch.abs(pooled - naive_pool(features, neighbors)) < 0.01).all()

    @pytest.mark.parametrize('config', TEST_CONFIGS_M_N_K_C)
    def test_cuda_pool(self, benchmark, config: tuple):
        from pool.cuda import cuda_pool
        features, neighbors, _ = generate_input_data(*config, device='cuda')
        pooled = benchmark(cuda_pool, features, neighbors)
        assert (torch.abs(pooled - naive_pool(features, neighbors)) < 0.01).all()

    #
    # # @pytest.mark.parametrize('m', M_VALUES)
    # # @pytest.mark.parametrize('n', N_VALUES)
    # # @pytest.mark.parametrize('k', K_VALUES)
    # # def test_sparse_pool(benchmark, n, m, k):
    # #     features, neighbors, _ = generate_input_data(n=n, m=m, k=k)
    # #     edges = SourceIndex(neighbors, dim_size=m).to_sparse_tensor()
    # #     conv = SimpleConv(aggr='max')
    # #     pooled = benchmark(conv, (features, None), edges, size=(m, n))
    # #     assert (torch.abs(pooled - indexed_pool(features, neighbors)) < 0.01).all()
    # #
    # #
    # # @pytest.mark.parametrize('m', M_VALUES)
    # # @pytest.mark.parametrize('n', N_VALUES)
    # # @pytest.mark.parametrize('k', K_VALUES)
    # # def test_tosparse_pool(benchmark, n, m, k):
    # #     features, neighbors, _ = generate_input_data(n=n, m=m, k=k)
    # #     conv = SimpleConv(aggr='max')
    # #
    # #     def sparse_pool(features, neighbors):
    # #         edges = SourceIndex(neighbors, dim_size=m).to_sparse_tensor()
    # #         return conv((features, None), edges, size=(m, n))
    # #
    # #     pooled = benchmark(sparse_pool, features, neighbors)
    # #     assert (torch.abs(pooled - indexed_pool(features, neighbors)) < 0.01).all()
    # #

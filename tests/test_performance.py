import pytest
import torch
from torch_geometric import EdgeIndex
from torch_geometric.nn import SimpleConv

from pool.naive import naive_pool

TEST_CONFIGS_M_N_K_C = [
    (8192 * 4, 8192, 32, 128),
]


def generate_input_data(m: int = 8192, n: int = 4096, k: int = 24, c: int = 128):
    features = torch.rand([m, c])
    neighbors = torch.randint(m, size=[n, k])
    edges = EdgeIndex(torch.stack([
        neighbors.flatten(),
        torch.arange(n).repeat_interleave(k),
    ]), sparse_size=(m, n)).validate()
    return features, neighbors, edges


@pytest.mark.parametrize('config', TEST_CONFIGS_M_N_K_C)
def test_naive_pool(benchmark, config):
    print(config)
    features, neighbors, _ = generate_input_data(*config)
    pass


# @pytest.mark.parametrize('m', M_VALUES)
# @pytest.mark.parametrize('n', N_VALUES)
# @pytest.mark.parametrize('k', K_VALUES)
# def test_indexed_pool(benchmark, n, m, k):
#     features, neighbors, _ = generate_input_data(n=n, m=m, k=k)
#     pooled = benchmark(naive_pool, features, neighbors)
#     assert pooled.shape == (n, features.size(-1))
#
#
# @pytest.mark.parametrize('m', M_VALUES)
# @pytest.mark.parametrize('n', N_VALUES)
# @pytest.mark.parametrize('k', K_VALUES)
# def test_pyg_pool(benchmark, n, m, k):
#     features, neighbors, edges = generate_input_data(n=n, m=m, k=k)
#     conv = SimpleConv(aggr='max')
#     pooled = benchmark(conv, (features, None), edges, size=(m, n))
#     assert (torch.abs(pooled - naive_pool(features, neighbors)) < 0.01).all()
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

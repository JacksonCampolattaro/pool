from typing import Callable

import pytest
import torch
from torch_geometric.testing.decorators import withDevice as parameterizeByDevice

from pool.naive import max_pool as naive_pool
from tests.util import VALUE_DTYPES, generate_input_data, edges_for_pool_function, POOL_FUNCTIONS

TEST_CONFIGS_M_N_K_C = [
    (1024, 128, 24, 32),
    (128, 1024, 24, 32),
    (1024 * 4, 1024 * 4, 24, 32),
    (1024 * 64, 1024 * 64, 24, 256),
    (1024 * 256, 1024 * 256, 24, 64),
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
        # torch.library.opcheck(pool_function, features, edges)

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

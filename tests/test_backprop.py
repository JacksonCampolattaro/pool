from typing import Callable

import torch
import pytest


from pool.naive import max_pool as naive_pool

from .test_inference import parameterizeByDevice, TEST_CONFIGS_M_N_K_C
from .util import VALUE_DTYPES, generate_input_data, edges_for_pool_function, POOL_FUNCTIONS

SMALL_TEST_CONFIGS_M_N_K_C = [
    (32, 32, 8, 16),
    (32, 64, 8, 16),
    (64, 32, 8, 16),
    (2048, 2048, 8, 16),
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
        # print(features)


        if dtype == torch.bfloat16:
            pytest.skip("BFloat not supported for backprop")

        # todo: way too lax
        # torch.autograd.gradcheck(pool_function, (features, edges), eps=0.1, atol=0.1, nondet_tol=0.1)

        out = pool_function(features, edges)
        out_ref = naive_pool(features_ref, neighbors)
        # print(out)

        (out ** 2).sum().backward()
        (out_ref ** 2).sum().backward()

        # print(features.grad)

        diff = features.grad - features_ref.grad
        close = torch.isclose(features.grad, features_ref.grad, rtol=1e-2)
        print(features_ref[~close], diff[~close], (~close).nonzero().size(0), features.size(0))
        # print((~close).float().mean())
        # print(features.grad - features_ref.grad)
        # print(torch.isclose(features.grad, features_ref.grad, atol=1e-4))
        # assert (~close).float().mean() < 0.001
        # print(features.grad - features_ref.grad)
        assert torch.isclose(features.grad, features_ref.grad, rtol=1e-2).all()

    @parameterizeByDevice
    @pytest.mark.parametrize('config', TEST_CONFIGS_M_N_K_C)
    @pytest.mark.parametrize('pool_function', POOL_FUNCTIONS)
    def test_roundtrip_speed(self, benchmark, device: str, config: tuple, pool_function: Callable):
        features, neighbors = generate_input_data(*config, device=device, requires_grad=True)
        edges = edges_for_pool_function(features, neighbors, pool_function)

        def fwd_and_bwd():
            pool_function(features, edges).sum().backward()

        benchmark(fwd_and_bwd)


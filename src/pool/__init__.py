from torch import Tensor

def max_pool(x: Tensor, index: Tensor):
    if x.is_cuda:
        from pool import cuda
        return cuda.max_pool(x, index)
    else:
        from pool import naive
        return naive.max_pool(x, index)

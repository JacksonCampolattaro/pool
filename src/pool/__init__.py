from torch import Tensor

def maxpool(x: Tensor, index: Tensor):
    if x.is_cuda:
        from . import cuda
        return cuda.maxpool(x, index)
    else:
        from . import naive
        return naive.maxpool(x, index)



from torch import Tensor

def maxpool(x: Tensor, index: Tensor):
    if x.is_cuda:
        from .cuda import maxpool as cuda_maxpool
        return cuda_maxpool(x, index)
    else:
        from .naive import maxpool as naive_maxpool
        return naive_maxpool(x, index)



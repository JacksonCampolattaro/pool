import os.path

import torch
from torch import Tensor
from torch.autograd import Function
from torch.amp import custom_fwd, custom_bwd
from torch.utils import cpp_extension

cuda = cpp_extension.load(
    'pool',
    sources=[
        os.path.join(os.path.dirname(__file__), 'maxpool.cu'),
    ],
    extra_cflags=["-O3", "-mavx2", "-funroll-loops"],
    extra_cuda_cflags=["-Xptxas", "-v"],
    verbose=True,
)

@torch.library.custom_op("pool::max_pool_infer", mutates_args=())
def max_pool_infer(x: Tensor, index: Tensor) -> Tensor:
    return cuda.maxpool_infer(x, index)
 
@torch.library.register_fake("pool::max_pool_infer")
def _(x, index):
    return x.new_empty([index.size(0), x.size(1)])

@torch.library.custom_op("pool::max_pool_forward", mutates_args=())
def max_pool_forward(x: Tensor, index: Tensor) -> tuple[Tensor, Tensor]:
    return cuda.maxpool_forward(x, index)
 
@torch.library.register_fake("pool::max_pool_forward")
def _(x, index):
    return x.new_empty([index.size(0), x.size(1)]), x.new_empty([index.size(0), x.size(1)], dtype=torch.int)

@torch.library.custom_op("pool::max_pool_backward", mutates_args=())
def max_pool_backward(m: int, indices: Tensor, grad: Tensor) -> Tensor:
    return cuda.maxpool_backward(m, indices, grad)
 
@torch.library.register_fake("pool::max_pool_backward")
def _(m, indices, grad):
    return grad.new_empty([m, grad.size(1)])


class Maxpool(Function):

    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        # assert x.is_cuda and index.is_cuda
        assert x.is_contiguous() and index.is_contiguous()
        assert index.dtype == torch.int64 # TODO: maybe other index types could be supported?
        if x.dtype == torch.half and x.shape[-1] % 8 != 0:
            raise ValueError("For float16, channel dimension must be a multiple of 8.")
        elif x.dtype == torch.float32 and x.shape[-1] % 4 != 0:
            raise ValueError("For float32, channel dimension must be a multiple of 4.")

        ctx.m = x.size(0)
        if ctx.needs_input_grad[0]:
            out, indices = max_pool_forward(x, index)
            ctx.save_for_backward(indices)
        else:
            out = max_pool_infer(x, index)

        return out

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad: torch.Tensor):
        grad = grad.contiguous()
        indices, = ctx.saved_tensors
        out = max_pool_backward(ctx.m, indices, grad)

        return out, None

# @torch.library.custom_op("pool::_max_pool", mutates_args=())
# def _max_pool(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
#     return Maxpool.apply(x, index)
#     # return torch.ops.pool.maxpool(x, index)

# @torch.library.register_fake("pool::_max_pool")
# def _(x, index):
#     return x.new_empty([index.size(0), x.size(1)], dtype=x.dtype)

max_pool = Maxpool.apply

# max_pool = cuda.maxpool

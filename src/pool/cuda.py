import os.path

import torch
from torch.autograd import Function
from torch.amp import custom_fwd, custom_bwd
from torch.utils import cpp_extension

cuda = cpp_extension.load(
    'maxpool_cuda_',
    sources=[
        os.path.join(os.path.dirname(__file__), 'maxpool.cu'),
    ],
    extra_cflags=["-O3", "-mavx2", "-funroll-loops"],
    extra_cuda_cflags=["-Xptxas", "-v"],
    verbose=True,
)


class Maxpool(Function):

    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        assert x.is_cuda and index.is_cuda
        assert x.is_contiguous() and index.is_contiguous()
        assert index.dtype == torch.int64 # TODO: maybe other index types could be supported?
        if x.dtype == torch.half and x.shape[-1] % 8 != 0:
            raise ValueError("For float16, channel dimension must be a multiple of 8.")
        elif x.dtype == torch.float32 and x.shape[-1] % 4 != 0:
            raise ValueError("For float32, channel dimension must be a multiple of 4.")


        out = torch.empty((index.size(0), x.size(-1)), dtype=x.dtype, device=x.device)

        ctx.m = x.size(0)
        if ctx.needs_input_grad[0]:
            indices = torch.empty_like(out, dtype=torch.uint32)
            cuda.maxpool_forward(out, indices, x, index)
            ctx.save_for_backward(indices)
        else:
            cuda.maxpool_infer(out, x, index)

        return out

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad: torch.Tensor):
        grad = grad.contiguous()
        out = torch.zeros((ctx.m, grad.size(-1)), dtype=grad.dtype, device=grad.device)
        indices, = ctx.saved_tensors
        cuda.maxpool_backward(out, indices, grad)
        return out, None


max_pool = Maxpool.apply

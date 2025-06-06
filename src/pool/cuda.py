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
    def forward(ctx, feature: torch.Tensor, knn: torch.Tensor, training: bool = True) -> torch.Tensor:
        assert feature.is_cuda and knn.is_cuda
        assert feature.is_contiguous() and knn.is_contiguous()
        assert knn.dtype == torch.int64
        if feature.dtype == torch.half:
            assert feature.shape[-1] % 8 == 0, \
                "16-bit cuda maxpool only supports channel dimensions which are multiples of 8"
        elif feature.dtype == torch.float32:
            assert feature.shape[-1] % 4 == 0, \
                "32-bit cuda maxpool only supports channel dimensions which are multiples of 4"

        output = torch.empty((knn.size(0), feature.size(-1)), dtype=feature.dtype, device=feature.device)
        if training or feature.requires_grad:
            indices = torch.empty_like(output, dtype=torch.uint32)
            cuda.maxpool_forward(output, indices, feature, knn)
            ctx.save_for_backward(indices)
        else:
            cuda.maxpool_infer(output, feature, knn)
        return output

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad: torch.Tensor):
        grad = grad.contiguous()
        output = -grad  # todo: why is this negated?
        indices, = ctx.saved_tensors
        cuda.maxpool_backward(output, indices, grad)
        return output, None, None


cuda_pool = Maxpool.apply

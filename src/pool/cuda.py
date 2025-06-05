import os.path

import torch
from torch.autograd import Function
from torch.amp import custom_fwd, custom_bwd
from torch.utils import cpp_extension

cutils = cpp_extension.load(
    'cuda_maxpool_',
    sources=[
        os.path.join(os.path.dirname(__file__), 'cuda_maxpool.cpp'),
        os.path.join(os.path.dirname(__file__), 'maxpool.cu'),
    ],
    extra_cflags=["-O3", "-mavx2", "-funroll-loops"],
    extra_cuda_cflags=["-Xptxas", "-v"],
    verbose=True,
)


class KEMP(Function):
    r"""
    f_i = max{f_j | j in knn_i} - f_i
    output = knn_edge_maxpooling(feature, knn, training=True)

    Only cuda version supported.

    feature: BNC, float / half
    knn:     BNk, int64
    output:  BNC, float / half

    While not training and gradient is not required,
    backward indices are not saved. Consumed time and space reduced slightly.
    """

    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, feature: torch.Tensor, knn: torch.Tensor, training: bool = True) -> torch.Tensor:
        assert feature.is_cuda and knn.is_cuda
        assert feature.is_contiguous() and knn.is_contiguous() and feature.size(0) == knn.size(0)
        assert knn.dtype == torch.int64
        if feature.dtype == torch.half:
            assert feature.shape[-1] % 8 == 0, "KEMP half precision impl only supports multiples of 8 as feature dim"
        elif feature.dtype == torch.float32:
            assert feature.shape[-1] % 4 == 0, "KEMP single precision impl only supports multiples of 4 as feature dim"
        else:
            raise NotImplementedError

        output = torch.empty((knn.size(0), feature.size(-1)), dtype=feature.dtype, device=feature.device)
        if training or feature.requires_grad:
            indices = torch.empty_like(output, dtype=torch.uint32)
            if feature.dtype == torch.half:
                cutils.half_aligned_knn_edge_maxpooling_forward(output, indices, feature, knn)
            else:
                cutils.aligned_knn_edge_maxpooling_forward(output, indices, feature, knn)
            ctx.save_for_backward(indices)
        else:
            if feature.dtype == torch.half:
                cutils.half_aligned_knn_edge_maxpooling_infer(output, feature, knn)
            else:
                cutils.aligned_knn_edge_maxpooling_infer(output, feature, knn)
        return output

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad: torch.Tensor):
        grad = grad.contiguous()
        output = -grad
        indices, = ctx.saved_tensors
        if grad.dtype == torch.half:
            cutils.half_knn_edge_maxpooling_backward(output, indices, grad)
        else:
            cutils.knn_edge_maxpooling_backward(output, indices, grad)
        return output, None, None


cuda_pool = KEMP.apply

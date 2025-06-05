#include <torch/extension.h>

void maxpool_backward(
    torch::Tensor &output,
    const torch::Tensor &indices,
    const torch::Tensor &grad
);

void maxpool_forward(
    torch::Tensor &output,
    torch::Tensor &indices,
    const torch::Tensor &feature,
    const torch::Tensor &knn
);

void maxpool_infer(
    torch::Tensor &output,
    const torch::Tensor &feature,
    const torch::Tensor &knn
);

void maxpool_backward(
    torch::Tensor &output,
    const torch::Tensor &indices,
    const torch::Tensor &grad
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("half_aligned_knn_edge_maxpooling_forward", &maxpool_forward);
    m.def("half_aligned_knn_edge_maxpooling_infer", &maxpool_infer);
    m.def("half_knn_edge_maxpooling_backward", &maxpool_backward);
}
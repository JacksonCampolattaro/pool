#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>
#include <assert.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

constexpr uint64_t group_size = 8;
constexpr uint64_t block = 512;

template<typename T>
struct __builtin_align__(group_size*sizeof(T)) value_group {
    T v[group_size];
};

// Specialization for half precision
using halfn = value_group<half>;

// Generic comparison function
template<typename T>
__device__ inline bool is_greater(const T& a, const T& b) {
    return a > b;
}

// Specialization for half precision
template<>
__device__ inline bool is_greater<half>(const half& a, const half& b) {
    return __hgt(a, b);
}

// Generic zero value
template<typename T>
__device__ inline T zero_value() {
    return T(0);
}

// Specialization for half precision
template<>
__device__ inline half zero_value<half>() {
    return __int2half_rz(0);
}

template<typename T>
__global__ void __launch_bounds__(block) maxpool_forward_kernel(
    T *output,
    uint32_t *indices,
    const T *feature,
    const uint64_t *knn,
    const uint64_t k,
    const uint64_t N,
    const uint64_t C_,
    const uint64_t NC
){
    // idx = bNC + nC + c
    const uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= NC) return;
    const uint64_t C = C_ / group_size;
    // bN + n
    const uint64_t BN = idx / C;
    const uint64_t n = BN % N;
    // feature base idx : bNC_ + c*group_size, striding C_
    const uint64_t feature_base = (BN - n) * C_ + (idx % C) * group_size;
    // knn base idx : bNk + nk, striding 1
    uint64_t knn_idx = BN * k;
    const uint64_t knn_end = knn_idx + k;
    uint64_t nbr_idx = knn[knn_idx];

    value_group<T> max_val = *(value_group<T>*)(feature + feature_base + nbr_idx * C_);
    uint32_t max_idx[group_size];
    for (uint64_t f_idx = 0; f_idx < group_size; ++f_idx) {
        max_idx[f_idx] = nbr_idx;
    }

    for (++knn_idx; knn_idx < knn_end; ++knn_idx) {
        nbr_idx = knn[knn_idx];
        const value_group<T> valn = *(value_group<T>*)(feature + feature_base + nbr_idx * C_);
        for (uint64_t f_idx = 0; f_idx < group_size; ++f_idx) {
            const T val = valn.v[f_idx];
            if (is_greater(val, max_val.v[f_idx])) {
                max_val.v[f_idx] = val;
                max_idx[f_idx] = nbr_idx;
            }
        }
    }

    const value_group<T> valn = *(value_group<T>*)(feature + feature_base + n * C_);
    for (uint64_t f_idx = 0; f_idx < group_size; ++f_idx) {
        indices[feature_base + n * C_ + f_idx] = max_idx[f_idx];
    }
    *(value_group<T>*)(output + feature_base + n * C_) = max_val;
}

void maxpool_forward(
    torch::Tensor &output,
    torch::Tensor &indices,
    const torch::Tensor &feature,
    const torch::Tensor &knn
){
    const uint64_t N = knn.size(0);
    const uint64_t k = knn.size(1);
    const uint64_t C = output.size(1);
    const uint64_t NC = 1 * N * (C / group_size);
    const uint64_t grid = (NC + block - 1) / block;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, feature.scalar_type(), "maxpool_forward", [&] {
        maxpool_forward_kernel<scalar_t><<<grid, block>>>(
            output.data_ptr<scalar_t>(),
            indices.data_ptr<uint32_t>(),
            feature.data_ptr<scalar_t>(),
            (const uint64_t*)knn.data_ptr(),
            k, N, C, NC
        );
    });
}

template<typename T>
__global__ void __launch_bounds__(block) maxpool_infer_kernel(
    T *output,
    const T *feature,
    const uint64_t *knn,
    const uint64_t k,
    const uint64_t N,
    const uint64_t C_,
    const uint64_t NC
){
    // idx = nC + c
    const uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= NC) return;
    const uint64_t C = C_ / group_size;
    // bN + n
    const uint64_t BN = idx / C;
    const uint64_t n = BN % N;
    // feature base idx : bNC_ + c*group_size, striding C_
    const uint64_t feature_base = (BN - n) * C_ + (idx % C) * group_size;
    // knn base idx : bNk + nk, striding 1
    uint64_t knn_idx = BN * k;
    const uint64_t knn_end = knn_idx + k;
    uint64_t nbr_idx = knn[knn_idx];

    value_group<T> max_val = *(value_group<T>*)(feature + feature_base + nbr_idx * C_);

    for (++knn_idx; knn_idx < knn_end; ++knn_idx) {
        nbr_idx = knn[knn_idx];
        const value_group<T> valn = *(value_group<T>*)(feature + feature_base + nbr_idx * C_);
        for (uint64_t f_idx = 0; f_idx < group_size; ++f_idx) {
            const T val = valn.v[f_idx];
            if (is_greater(val, max_val.v[f_idx])) {
                max_val.v[f_idx] = val;
            }
        }
    }

    *(value_group<T>*)(output + feature_base + n * C_) = max_val;
}

void maxpool_infer(
    torch::Tensor &output,
    const torch::Tensor &feature,
    const torch::Tensor &knn
){
    const uint64_t k = knn.size(1);
    const uint64_t N = knn.size(0);
    const uint64_t C = output.size(1);
    const uint64_t NC = 1 * N * (C / group_size);
    const uint64_t grid = (NC + block - 1) / block;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, feature.scalar_type(), "maxpool_infer", [&] {
        maxpool_infer_kernel<scalar_t><<<grid, block>>>(
            output.data_ptr<scalar_t>(),
            feature.data_ptr<scalar_t>(),
            (const uint64_t*)knn.data_ptr(),
            k, N, C, NC
        );
    });
}

// todo: this is almost certainly incorrect; I need a unit test
template<typename T>
__global__ void maxpool_backward_kernel(
    T *output,
    const uint32_t *indices,
    const T *grad,
    const uint64_t N,
    const uint64_t C,
    const uint64_t NC
){
    const uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= NC) return;
    const uint64_t n = idx / C % N;
    const uint64_t backidx = indices[idx];
    const T g = grad[idx];
    const uint64_t feature_base = idx - n*C + backidx*C;

    // Generic atomic add - may need specialization for different types
    atomicAdd(output + feature_base, g);
}

template<>
__global__ void maxpool_backward_kernel<at::Half>(
    at::Half *output,
    const uint32_t *indices,
    const at::Half *grad,
    const uint64_t N,
    const uint64_t C,
    const uint64_t NC
){
    const uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= NC) return;
    const uint64_t n = idx / C % N;
    const uint64_t backidx = indices[idx];
    const half g = grad[idx];
    const uint64_t high = idx % 2;
    const uint64_t feature_base = idx - n*C + backidx*C - high;

    half2 x;
    x.x = high ? __int2half_rz(0) : g;
    x.y = high ? g : __int2half_rz(0);
    atomicAdd(reinterpret_cast<half2*>(output + feature_base), x);
}

void maxpool_backward(
    torch::Tensor &output,
    const torch::Tensor &indices,
    const torch::Tensor &grad
){
    const uint64_t M = indices.size(0);
    const uint64_t C = output.size(1);
    const uint64_t MC = M * C;
    const uint64_t grid = (MC + block - 1) / block;

    AT_DISPATCH_FLOATING_TYPES_AND(at::kHalf, output.scalar_type(), "maxpool_backward", [&] {
        maxpool_backward_kernel<scalar_t><<<grid, block>>>(
            output.data_ptr<scalar_t>(),
            indices.data_ptr<uint32_t>(),
            grad.data_ptr<scalar_t>(),
            M, C, MC
        );
    });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("maxpool_forward", &maxpool_forward);
    m.def("maxpool_infer", &maxpool_infer);
    m.def("maxpool_backward", &maxpool_backward);
}

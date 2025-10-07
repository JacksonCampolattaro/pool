#include <assert.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <torch/library.h>

using torch::Tensor;



constexpr uint64_t group_size = 8;
constexpr uint64_t block = 512;

template <typename T>
struct __builtin_align__(group_size * sizeof(T)) value_group {
  T v[group_size];
};

// Generic comparison function
template <typename T>
__device__ inline bool is_greater(const T &a, const T &b) {
  return a > b;
}

// Specialization for half precision
template <>
__device__ inline bool is_greater<half>(const half &a, const half &b) {
  return __hgt(a, b);
}

// Generic zero value
template <typename T> __device__ inline T zero_value() { return T(0); }

// Specialization for half precision
template <> __device__ inline half zero_value<half>() {
  return __int2half_rz(0);
}

template <typename T>
__global__ void __launch_bounds__(block)
    maxpool_forward_kernel(T *output, uint32_t *indices, const T *feature,
                           const uint64_t *knn, const uint64_t k,
                           const uint64_t N, const uint64_t C_,
                           const uint64_t NC) {
  // idx = bNC + nC + c
  const uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= NC)
    return;
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

  value_group<T> max_val =
      *(value_group<T> *)(feature + feature_base + nbr_idx * C_);
  uint32_t max_idx[group_size];
  for (uint64_t f_idx = 0; f_idx < group_size; ++f_idx) {
    max_idx[f_idx] = nbr_idx;
  }

  for (++knn_idx; knn_idx < knn_end; ++knn_idx) {
    nbr_idx = knn[knn_idx];
    const value_group<T> valn =
        *(value_group<T> *)(feature + feature_base + nbr_idx * C_);
    for (uint64_t f_idx = 0; f_idx < group_size; ++f_idx) {
      const T val = valn.v[f_idx];
      if (is_greater(val, max_val.v[f_idx])) {
        max_val.v[f_idx] = val;
        max_idx[f_idx] = nbr_idx;
      }
    }
  }

  const value_group<T> valn =
      *(value_group<T> *)(feature + feature_base + n * C_);
  for (uint64_t f_idx = 0; f_idx < group_size; ++f_idx) {
    indices[feature_base + n * C_ + f_idx] = max_idx[f_idx];
  }

  *(value_group<T> *)(output + feature_base + n * C_) = max_val;
  //    const uint64_t output_base = feature_base + n * C_;
  //    if constexpr (std::is_same_v<T, at::Half>) {
  //        // Special handling for half precision
  //        for (uint64_t f_idx = 0; f_idx < group_size/2; ++f_idx)
  //            *reinterpret_cast<half2*>(output + output_base + f_idx*2) =
  //                __halves2half2(max_val.v[f_idx*2], max_val.v[f_idx*2+1]);
  //
  //    } else {
  //        // Default handling for other types
  //        *(value_group<T>*)(output + output_base) = max_val;
  //    }
}

void maxpool_forward_inplace(Tensor &output, Tensor &indices,
                             const Tensor &feature, const Tensor &knn) {
  const uint64_t N = knn.size(0);
  const uint64_t k = knn.size(1);
  const uint64_t C = output.size(1);
  const uint64_t NC = 1 * N * (C / group_size);
  const uint64_t grid = (NC + block - 1) / block;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, feature.scalar_type(), "maxpool_forward", [&] {
        maxpool_forward_kernel<scalar_t><<<grid, block>>>(
            output.data_ptr<scalar_t>(), indices.data_ptr<uint32_t>(),
            feature.data_ptr<scalar_t>(), (const uint64_t *)knn.data_ptr(), k,
            N, C, NC);
      });
}

std::tuple<Tensor, Tensor> maxpool_forward(const Tensor &feature,
                                           const Tensor &knn) {
  const int64_t N = knn.size(0);
  const int64_t C = feature.size(1);

  auto output = torch::empty({N, C}, feature.options());
  auto indices = torch::empty(
      {N, C}, torch::dtype(torch::kUInt32).device(feature.device()));

  maxpool_forward_inplace(output, indices, feature, knn);
  return {output, indices};
}

template <typename T>
__global__ void __launch_bounds__(block)
    maxpool_infer_kernel(T *output, const T *feature, const uint64_t *knn,
                         const uint64_t k, const uint64_t N, const uint64_t C_,
                         const uint64_t NC) {
  // idx = nC + c
  const uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= NC)
    return;
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

  value_group<T> max_val =
      *(value_group<T> *)(feature + feature_base + nbr_idx * C_);

  for (++knn_idx; knn_idx < knn_end; ++knn_idx) {
    nbr_idx = knn[knn_idx];
    const value_group<T> valn =
        *(value_group<T> *)(feature + feature_base + nbr_idx * C_);
    for (uint64_t f_idx = 0; f_idx < group_size; ++f_idx) {
      const T val = valn.v[f_idx];
      if (is_greater(val, max_val.v[f_idx])) {
        max_val.v[f_idx] = val;
      }
    }
  }

  *(value_group<T> *)(output + feature_base + n * C_) = max_val;

  //    const uint64_t output_base = feature_base + n * C_;
  //    if constexpr (std::is_same_v<T, at::Half>) {
  //        for (uint64_t f_idx = 0; f_idx < group_size/2; ++f_idx)
  //            *reinterpret_cast<half2*>(output + output_base + f_idx*2) =
  //                __halves2half2(max_val.v[f_idx*2], max_val.v[f_idx*2+1]);
  //
  //    } else {
  //        *(value_group<T>*)(output + output_base) = max_val;
  //    }
}

void maxpool_infer_inplace(Tensor &output, const Tensor &feature,
                           const Tensor &knn) {
  const uint64_t k = knn.size(1);
  const uint64_t N = knn.size(0);
  const uint64_t C = output.size(1);
  const uint64_t NC = 1 * N * (C / group_size);
  const uint64_t grid = (NC + block - 1) / block;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, feature.scalar_type(), "maxpool_infer", [&] {
        maxpool_infer_kernel<scalar_t><<<grid, block>>>(
            output.data_ptr<scalar_t>(), feature.data_ptr<scalar_t>(),
            (const uint64_t *)knn.data_ptr(), k, N, C, NC);
      });
}

Tensor maxpool_infer(const Tensor &feature, const Tensor &knn) {
  const int64_t N = knn.size(0);
  const int64_t C = feature.size(1);

  auto output = torch::empty({N, C}, feature.options());

  maxpool_infer_inplace(output, feature, knn);
  return output;
}

template <typename T>
__global__ void maxpool_backward_kernel(T *output, const uint32_t *indices,
                                        const T *grad, const uint64_t N,
                                        const uint64_t C, const uint64_t NC) {
  const uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= NC)
    return;
  const uint64_t n = idx / C % N;
  const uint64_t backidx = indices[idx];
  const T g = grad[idx];
  const uint64_t feature_base = idx - n * C + backidx * C;

  if constexpr (std::is_same_v<T, at::Half>) {
    const uint64_t high = idx % 2;
    const half2 x = __halves2half2(
        high ? __int2half_rz(0) : *reinterpret_cast<const half *>(&g),
        high ? *reinterpret_cast<const half *>(&g) : __int2half_rz(0));
    atomicAdd(reinterpret_cast<half2 *>(output + feature_base - high), x);
  } else {
    atomicAdd(output + feature_base, g);
  }
}

void maxpool_backward_inplace(Tensor &output, const Tensor &indices,
                              const Tensor &grad) {
  const uint64_t M = indices.size(0);
  const uint64_t C = output.size(1);
  const uint64_t MC = M * C;
  const uint64_t grid = (MC + block - 1) / block;

  AT_DISPATCH_FLOATING_TYPES_AND(
      at::kHalf, output.scalar_type(), "maxpool_backward", [&] {
        maxpool_backward_kernel<scalar_t><<<grid, block>>>(
            output.data_ptr<scalar_t>(), indices.data_ptr<uint32_t>(),
            grad.data_ptr<scalar_t>(), M, C, MC);
      });
}

Tensor maxpool_backward(int64_t m, const Tensor &indices, const Tensor &grad) {
  const int64_t C = grad.size(1);
  auto output = torch::zeros({m, C}, grad.options());
  maxpool_backward_inplace(output, indices, grad);
  return output;
}

class MaxPool : public torch::autograd::Function<MaxPool> {
    public:

        static Tensor forward(
            torch::autograd::AutogradContext* ctx,
            const Tensor &x,
            const Tensor &index
        ) {
            
            // todo: check DTypes
            auto x_contig = x.contiguous();
            auto index_contig = index.contiguous();            

            ctx->saved_data["m"] = x.size(0);
            auto [out, indices] = maxpool_forward(x_contig, index_contig);
            ctx->save_for_backward({indices});
            return out;
            // if (ctx->needs_input_grad(0)) {
            //     auto [out, indices] = maxpool_forward(x_contig,
            //     index_contig); ctx->save_for_backward({indices}); return out;
            // } else {
            //     return maxpool_infer(x_contig, index_contig);
            // }
        }

        static torch::autograd::variable_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::variable_list grad
        ) {

            auto grad_contig = grad[0].contiguous();
            auto indices = ctx->get_saved_variables()[0];
            auto m = ctx->saved_data["m"].toInt();
            return {maxpool_backward(m, indices, grad_contig), {}};
        }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("maxpool_forward_inplace", &maxpool_forward_inplace);
  // m.def("maxpool_infer_inplace", &maxpool_infer_inplace);
  // m.def("maxpool_backward_inplace", &maxpool_backward_inplace);
  //   m.def("maxpool", [](const Tensor &x, const Tensor &index) -> Tensor {
  //   return MaxPool::apply(x, index);
  // });
  m.def("maxpool_forward", &maxpool_forward);
  m.def("maxpool_infer", &maxpool_infer);
  m.def("maxpool_backward", &maxpool_backward);
}

TORCH_LIBRARY(pool, m) {
  m.def("maxpool(Tensor x, Tensor index) -> Tensor");
}

TORCH_LIBRARY_IMPL(pool, Autograd, m) {
  m.impl("maxpool", [](const Tensor &x, const Tensor &index) -> Tensor {
    return MaxPool::apply(x, index);
  });
}

TORCH_LIBRARY_IMPL(pool, CUDA, m) {
  m.impl("maxpool", [](const Tensor &x, const Tensor &index) -> Tensor {
    return maxpool_infer(x, index);
  });
}

TORCH_LIBRARY_IMPL(pool, Meta, m) {
  m.impl("maxpool", [](const Tensor &x, const Tensor &index) -> Tensor {
    auto N = index.size(0);
    auto C = x.size(1);
    return at::empty({N, C}, x.options());
  });
}


// TORCH_LIBRARY_IMPL(cuda_maxpool, Meta, m) {
//   m.impl("maxpool", [](const Tensor &x, const Tensor &index) -> Tensor {
//     auto N = index.size(0);
//     auto C = x.size(1);
//     return at::empty({N, C}, x.options());
//   });
// }

// TORCH_LIBRARY_IMPL(cuda_maxpool, Meta, m) {
//   m.impl("maxpool", [](const Tensor &feature, const Tensor &knn) -> Tensor {
//     return MaxPool::apply(feature, knn);
//   });
// }

// TORCH_LIBRARY(cuda_maxpool, m) {
//   // m.def("maxpool_forward_inplace", &maxpool_forward_inplace);
//   // m.def("maxpool_infer_inplace", &maxpool_infer_inplace);
//   // m.def("maxpool_backward_inplace", &maxpool_backward_inplace);
//   m.def("maxpool_forward", &maxpool_forward);
//   m.def("maxpool_infer", &maxpool_infer);
//   m.def("maxpool_backward", &maxpool_backward);
// }

// TORCH_LIBRARY_IMPL(cuda_maxpool, Meta, m) {

//   // m.impl("maxpool_forward_inplace",
//   //        [](Tensor &out, Tensor &indices, const Tensor &feature,
//   //           const Tensor &knn) {
//   //          // Shape inference
//   //          auto N = knn.size(0);
//   //          auto C = feature.size(1);
//   //          out.resize_({N, C});
//   //          indices.resize_({N, C}).toType(torch::kUInt32);
//   //        });

//   // m.impl("maxpool_infer_inplace",
//   //        [](Tensor &out, const Tensor &feature, const Tensor &knn) {
//   //          auto N = knn.size(0);
//   //          auto C = feature.size(1);
//   //          out.resize_({N, C});
//   //        });

//   // m.impl(
//   //     "maxpool_backward_inplace",
//   //     [](Tensor &grad_input, const Tensor &indices, const Tensor &grad_output) {
//   //       auto M = grad_input.size(0);
//   //       auto C = grad_output.size(1);
//   //       grad_input.resize_({M, C});
//   //     });

//   m.impl("maxpool_forward",
//          [](const Tensor &feature,
//             const Tensor &knn) -> std::tuple<Tensor, Tensor> {
//            auto N = knn.size(0);
//            auto C = feature.size(1);
//            auto output = at::empty({N, C}, feature.options());
//            auto indices = at::empty({N, C}, feature.options().dtype(at::kInt));
//            return std::make_tuple(output, indices);
//          });

//   // Non-inplace infer (returns a new tensor)
//   m.impl("maxpool_infer",
//          [](const Tensor &feature, const Tensor &knn) -> Tensor {
//            auto N = knn.size(0);
//            auto C = feature.size(1);
//            return at::empty({N, C}, feature.options());
//          });

//   // Non-inplace backward (returns a new tensor)
//   m.impl("maxpool_backward",
//          [](int64_t m, const Tensor &indices, const Tensor &grad) -> Tensor {
//            auto C = indices.size(1);
//            return at::empty({m, C}, grad.options());
//          });
// }

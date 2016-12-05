#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/more_math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void maximum_kernel(const int N, const Dtype* const X,
                               Dtype* const Y, const int s, const int e) {
  CUDA_KERNEL_LOOP(index, N) {
    Y[index] = X[index];
    for (int i = index + s; i < e; i += s) {
      if (Y[index] < X[i]) {
        Y[i] = X[i];
      }
    }
  }
}

template <typename Dtype>
void caffe_gpu_maximum(const int N, const Dtype* const X, Dtype* const Y,
                       const int s, const int e) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  maximum_kernel<Dtype> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>
      (N, X, Y, s, e);
}

template void caffe_gpu_maximum<int>(const int N, const int* const X,
                                     int* const Y, const int s, const int e);
template void caffe_gpu_maximum<float>(const int N, const float* const X,
                                       float* const Y, const int s,
                                       const int e);
template void caffe_gpu_maximum<double>(const int N, const double* const X,
                                        double* const Y, const int s,
                                        const int e);

template <typename Dtype>
__global__ void or_kernel(const int N, const Dtype* const x,
                          const Dtype* const y, Dtype* const z) {
  CUDA_KERNEL_LOOP(index, N) {
    if (x[index] == 1 || y[index] == 1)
      z[index] = 1;
    else
      z[index] = 0;
  }
}

template <typename Dtype>
void caffe_gpu_or(const int N, const Dtype* const x, const Dtype* const y,
                  Dtype* const z) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  or_kernel<Dtype> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>
      (N, x, y, z);
}

template void caffe_gpu_or<int>(const int N, const int* const x,
                                const int* const y, int* z);
template void caffe_gpu_or<float>(const int N, const float* const x,
                                  const float* const y, float* z);
template void caffe_gpu_or<double>(const int N, const double* const x,
                                   const double* const y, double* z);

template <typename Dtype>
__global__ void ceil_kernel(const int N, Dtype* const x) {
  CUDA_KERNEL_LOOP(index, N) { x[index] = ceil(x[index]); }
}

template <typename Dtype>
void caffe_gpu_ceil(const int N, Dtype* const x) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  ceil_kernel<Dtype> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>> (N, x);
}

template void caffe_gpu_ceil<float>(const int N, float* const x);
template void caffe_gpu_ceil<double>(const int N, double* const x);

template <typename Dtype>
__global__ void floor_kernel(const int n, Dtype* const x) {
  CUDA_KERNEL_LOOP(index, n) { x[index] = floor(x[index]); }
}

template <typename Dtype>
void caffe_gpu_floor(const int N, Dtype* const x) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  floor_kernel<Dtype> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>> (N, x);
}

template void caffe_gpu_floor<float>(const int n, float* const x);
template void caffe_gpu_floor<double>(const int n, double* const x);

template <typename Dtype>
__global__ void without_kernel(const int N, Dtype* const x, const Dtype without,
                               const Dtype replace) {
  CUDA_KERNEL_LOOP(index, N) {
    if (x[index] == without) x[index] = replace;
  }
}

template <typename Dtype>
void caffe_gpu_without(const int N, Dtype* const x, const Dtype without,
                       const Dtype replace) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  without_kernel<Dtype> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>
      (N, x, without, replace);
}

template void caffe_gpu_without<float>(const int N, float* const x,
                                       const float without,
                                       const float replace);
template void caffe_gpu_without<double>(const int N, double* const x,
                                        const double without,
                                        const double replace);

}  // namespace caffe

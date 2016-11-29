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
void caffe_gpu_or(const int N, const Dtype* x, const Dtype* y, Dtype* z) {}

template void caffe_gpu_or<int>(const int N, const int* x, const int* y,
                                int* z);
template void caffe_gpu_or<float>(const int N, const float* x, const float* y,
                                  float* z);
template void caffe_gpu_or<double>(const int N, const double* x,
                                   const double* y, double* z);

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

template void caffe_gpu_maximum(const int N, const int* const X,
                                int* const Y, const int s, const int e);
template void caffe_gpu_maximum(const int N, const float* const X,
                                float* const Y, const int s, const int e);
template void caffe_gpu_maximum(const int N, const double* const X,
                                double* const Y, const int s, const int e);

}  // namespace caffe

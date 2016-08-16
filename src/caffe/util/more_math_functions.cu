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
void caffe_gpu_or(const int N, const Dtype *x,const Dtype* y,Dtype* z) {
}

template void caffe_gpu_or<int>(const int N, const int *x, const int* y,int* z);
template void caffe_gpu_or<float>(const int N, const float *x, const float* y,float* z);
template void caffe_gpu_or<double>(const int N, const double *x, const double* y,double* z);

}  // namespace caffe

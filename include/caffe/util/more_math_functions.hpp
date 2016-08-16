#ifndef CAFFE_UTIL_MORE_MATH_FUNCTION_H_
#define CAFFE_UTIL_MORE_MATH_FUNCTION_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit

#include "glog/logging.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"

namespace caffe {

template <typename Dtype>
Dtype caffe_cpu_max_element(const int n, const Dtype* x);

template <typename Dtype>
int caffe_cpu_threshold_bbox(Blob<Dtype> *cpg_blob, Blob<Dtype> *bboxes_blob,
                         const float fg_threshold, const int gt_label);

#ifndef CPU_ONLY  // GPU

template <typename Dtype>
void caffe_gpu_or(const int n, const Dtype* x, const Dtype* y,Dtype* z);

#endif  // !CPU_ONLY

}  // namespace caffe

#endif  // CAFFE_UTIL_MORE_MATH_FUNCTIONS_H_

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/mil_ms_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void filter_kernel(const int N, const Dtype* const in_filter,
                              const int num_roi, const int num_class,
                              Dtype* const filter) {
  CUDA_KERNEL_LOOP(index, N) {
    const int n = index / num_class;
    const int c = index % num_class;
    if (in_filter[n * num_class + c] == 1) {
      for (int cc = 0; cc < num_class; ++cc) {
        filter[c * num_roi * num_class + n * num_class + cc] = 1;
      }

    } else {
      for (int cc = 0; cc < num_class; ++cc) {
        filter[c * num_roi * num_class + n * num_class + cc] = 0;
      }
    }
  }
}

template <typename Dtype>
void MIL_MSLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  filter_kernel<Dtype> << <CAFFE_GET_BLOCKS(num_class_ * num_roi_),
                           CAFFE_CUDA_NUM_THREADS>>>
      (num_class_ * num_roi_, bottom[1]->gpu_data(), num_roi_, num_class_,
       filter_.mutable_gpu_data());
  for (int c = 0; c < num_class_; ++c) {
    caffe_gpu_mul(num_spatial_, filter_.gpu_data() + c * num_spatial_,
                  bottom[0]->mutable_gpu_data(),
                  top[0]->mutable_gpu_data() + c * num_spatial_);
  }
}

template <typename Dtype>
void MIL_MSLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << "filter is not backprapagated";
  }

  if (propagate_down[0]) {
    for (int c = 0; c < num_class_; ++c) {
      caffe_gpu_mul(
          num_roi_ * num_class_, filter_.gpu_data() + c * num_spatial_,
          top[0]->gpu_diff() + c * num_spatial_, bottom[0]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MIL_MSLayer);

}  // namespace caffe

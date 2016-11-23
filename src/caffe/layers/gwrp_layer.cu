#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/gwrp_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void gwrp_forward_gpu(const int num_class, const int num_roi,
                                 const Dtype normalization,
                                 const Dtype *const weight_data,
                                 Dtype *const rank_data,
                                 Dtype *const rank_id_data,
                                 Dtype *const top_data) {
  CUDA_KERNEL_LOOP(index, num_class) {
    // init rank_id_data
    for (int i = 0; i < num_roi; ++i) {
      int ii = index + num_class * i;
      rank_id_data[ii] = i;
    }
    for (int i = 0; i < num_roi; ++i) {
      int ii = index + num_class * i;
      for (int j = i + 1; j < num_roi; ++j) {
        int jj = index + num_class * j;
        if (rank_data[ii] < rank_data[jj]) {
          // swap value and index
          Dtype tmp_value = rank_data[ii];
          rank_data[ii] = rank_data[jj];
          rank_data[jj] = tmp_value;

          int tmp_id = rank_id_data[ii];
          rank_id_data[ii] = rank_id_data[jj];
          rank_id_data[jj] = tmp_id;
        }
      }
    }
    Dtype sum = 0;
    for (int i = 0; i < num_roi; ++i) {
      int ii = index + num_class * i;
      sum += rank_data[ii] * weight_data[i];
    }
    top_data[index] = sum / normalization;
  }
}

template <typename Dtype>
void GWRPLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                   const vector<Blob<Dtype> *> &top) {
  caffe_copy(count_, bottom[0]->gpu_data(), bottom[0]->mutable_gpu_diff());
  Dtype normalization = weight_sum_.cpu_data()[num_roi_];

  // NOLINT_NEXT_LINE(whitespace/operators)
  gwrp_forward_gpu<Dtype> << <CAFFE_GET_BLOCKS(num_class_),
                              CAFFE_CUDA_NUM_THREADS>>>
      (num_class_, num_roi_, normalization, weight_.cpu_data(),
       bottom[0]->mutable_gpu_diff(), rank_id_.mutable_gpu_data(),
       top[0]->mutable_gpu_data());
}

template <typename Dtype>
__global__ void gwrp_backward_gpu(const int num_class, const int num_roi,
                                  const Dtype normalization,
                                  const Dtype *const rank_id_data,
                                  const Dtype *const weight_data,
                                  const Dtype *const top_diff,
                                  Dtype *const bottom_diff) {
  CUDA_KERNEL_LOOP(index, num_class) {
    for (int i = 0; i < num_roi; ++i) {
      int ii = index + num_class * i;
      int rank_id = rank_id_data[ii];
      bottom_diff[index + num_class * rank_id] =
          1.0 * top_diff[index] * weight_data[i] / normalization;
    }
  }
}

template <typename Dtype>
void GWRPLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
                                    const vector<bool> &propagate_down,
                                    const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[0] == false) {
    return;
  }

  Dtype normalization = weight_sum_.cpu_data()[num_roi_];

  // NOLINT_NEXT_LINE(whitespace/operators)
  gwrp_backward_gpu<Dtype> << <CAFFE_GET_BLOCKS(num_class_),
                               CAFFE_CUDA_NUM_THREADS>>>
      (num_class_, num_roi_, normalization, rank_id_.gpu_data(),
       weight_.gpu_data(), top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(GWRPLayer);

}  // namespace caffe

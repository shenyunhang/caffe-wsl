#include <vector>

#include "caffe/layers/polar_constraint_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ceil_gpu(const int count, Dtype* const data) {
  CUDA_KERNEL_LOOP(index, count) { data[index] = ceil(data[index]); }
}

template <typename Dtype>
__global__ void floor_gpu(const int count, Dtype* const data) {
  CUDA_KERNEL_LOOP(index, count) { data[index] = floor(data[index]); }
}

template <typename Dtype>
__global__ void without_gpu(const int count, Dtype* const data,
                            const Dtype without, const Dtype replace) {
  CUDA_KERNEL_LOOP(index, count) {
    if (data[index] == without) data[index] = replace;
  }
}

template <typename Dtype>
void PolarConstraintLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // data 	#roi 	#class 	1 	1
  // fliter	#roi 	#class 	1 	1
  // [0,1) denote the score
  // 1 denote in postive bag

  caffe_copy(count_, bottom[1]->gpu_data(), fliter.mutable_gpu_data());
  // NOLINT_NEXT_LINE(whitespace/operators)
  floor_gpu<Dtype><<<CAFFE_GET_BLOCKS(count_), CAFFE_CUDA_NUM_THREADS>>>(
      count_, fliter.mutable_gpu_data());

  if (polar_) {
  } else {
    // 翻天覆地
    caffe_gpu_scal(count_, Dtype(-1), fliter.mutable_gpu_data());
    caffe_gpu_add_scalar(count_, Dtype(1), fliter.mutable_gpu_data());
  }
  caffe_gpu_mul(count_, fliter.gpu_data(), bottom[0]->gpu_data(),
                top[0]->mutable_gpu_data());
  if (polar_) {
  } else {
    /*caffe_gpu_set(channels_, Dtype(0), top[1]->mutable_gpu_data());*/

    /*caffe_copy(count_, bottom[1]->gpu_data(), top[1]->mutable_gpu_data());*/
    /*// NOLINT_NEXT_LINE(whitespace/operators)*/
    /*without_gpu<Dtype><<<CAFFE_GET_BLOCKS(count_),
     * CAFFE_CUDA_NUM_THREADS>>>(*/
    /*count_, top[1]->mutable_gpu_data(), Dtype(1), Dtype(0));*/
  }
}

template <typename Dtype>
void PolarConstraintLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to second inputs.";
  }

  if (propagate_down[0]) {
    caffe_gpu_mul(count_, fliter.gpu_data(), top[0]->gpu_diff(),
                  bottom[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PolarConstraintLayer);

}  // namespace caffe

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/ya_softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void rm_ignore_label_gpu(const int count, const Dtype* const label,
                                    Dtype* const in) {
  CUDA_KERNEL_LOOP(index, count) {
    if (in[index] == -1) in[index] = label[index];
  }
}

template <typename Dtype>
void YASoftmaxWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  /*// NOLINT_NEXT_LINE(whitespace/operators)*/
  /*rm_ignore_label_gpu<Dtype> << <CAFFE_GET_BLOCKS(bottom[0]->count()),*/
  /*CAFFE_CUDA_NUM_THREADS>>>*/
  /*(bottom[0]->count(), bottom[1]->gpu_data(),*/
  /*bottom[0]->mutable_gpu_data());*/
  softmax_loss_layer_->Forward(softmax_loss_bottom_vec_, softmax_loss_top_vec_);

  // -------------------------------------------------------------------------
  total_iter_++;
  total_sample_ += bottom[1]->count();
  total_loss_ += top[0]->cpu_data()[0];
  accum_iter_++;
  accum_sample_ += bottom[1]->count();
  accum_loss_ += top[0]->cpu_data()[0];
  if (accum_iter_ == 1280) {
    /*LOG(INFO) << this->layer_param().name() << " #iter: " << total_iter_*/
              /*<< "#sample : " << total_sample_ << " #loss: " << total_loss_*/
              /*<< " AVE loss: " << 1.0 * total_loss_ / total_iter_;*/

    LOG(INFO) << this->layer_param().name() << " #iter: " << accum_iter_
              << " #sample: " << accum_sample_ << " #loss: " << accum_loss_
              << " AVE loss: " << 1.0 * accum_loss_ / accum_iter_;
    accum_iter_ = 0;
    accum_sample_ = 0;
    accum_loss_ = 0;
  }
}

template <typename Dtype>
void YASoftmaxWithLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    softmax_loss_layer_->Backward(softmax_loss_top_vec_, propagate_down,
                                  softmax_loss_bottom_vec_);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(YASoftmaxWithLossLayer);

}  // namespace caffe

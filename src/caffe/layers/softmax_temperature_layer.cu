#include <vector>

#include "caffe/layers/softmax_temperature_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxTemperatureLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (is_append_)
    Copy_blob(bottom[0], softmax_input_.get(), false);
  else
    softmax_input_->CopyFrom(*bottom[0], false, false);
  caffe_gpu_scal(softmax_input_->count(), temperature_,
                 softmax_input_->mutable_gpu_data());
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  if (is_append_)
    Copy_blob(softmax_output_.get(), top[0], false);
  else
    top[0]->CopyFrom(*softmax_output_, false, false);
}

template <typename Dtype>
void SoftmaxTemperatureLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (is_append_)
    Copy_blob(top[0], softmax_output_.get(), true);
  else
    softmax_output_->CopyFrom(*top[0], true, false);
  softmax_layer_->Backward(softmax_top_vec_, propagate_down,
                           softmax_bottom_vec_);
  caffe_gpu_scal(softmax_input_->count(), temperature_,
                 softmax_input_->mutable_gpu_diff());
  if (is_append_)
    Copy_blob(softmax_input_.get(), bottom[0], true);
  else
    bottom[0]->CopyFrom(*softmax_input_, true, false);
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxTemperatureLayer);

}  // namespace caffe

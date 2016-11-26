#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/ya_softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void YASoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_loss_param(this->layer_param_);
  softmax_loss_param.set_type("SoftmaxWithLoss");
  softmax_loss_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_loss_param);
  softmax_loss_bottom_vec_.clear();
  softmax_loss_bottom_vec_.push_back(bottom[0]);
  softmax_loss_bottom_vec_.push_back(bottom[1]);
  softmax_loss_top_vec_.clear();
  softmax_loss_top_vec_.push_back(top[0]);
  if (top.size() > 1) {
    softmax_loss_top_vec_.push_back(top[1]);
  }
  softmax_loss_layer_->SetUp(softmax_loss_bottom_vec_, softmax_loss_top_vec_);
}

template <typename Dtype>
void rm_ignore_label(const int count, const Dtype* const label,
                     Dtype* const in) {
  for (int i = 0; i < count; ++i) {
    if (in[i] == -1) in[i] = label[i];
  }
}

template <typename Dtype>
void YASoftmaxWithLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_loss_layer_->Reshape(softmax_loss_bottom_vec_, softmax_loss_top_vec_);
}

template <typename Dtype>
void YASoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //rm_ignore_label(bottom[0]->count(), bottom[1]->cpu_data(),
                  //bottom[0]->mutable_cpu_data());
  // The forward pass computes the softmax loss values.
  softmax_loss_layer_->Forward(softmax_loss_bottom_vec_, softmax_loss_top_vec_);

  // -------------------------------------------------------------------------
  total_iter_++;
  total_sample_ += bottom[1]->count();
  total_loss_ += top[0]->cpu_data()[0];
  accum_iter_++;
  accum_sample_ += bottom[1]->count();
  accum_loss_ += top[0]->cpu_data()[0];
  if (accum_iter_ == 1280) {
    LOG(INFO) << "#iter: " << total_iter_ << " #sample: " << total_sample_
              << " #loss: " << total_loss_
              << " AVE loss: " << 1.0 * total_loss_ / total_iter_;

    LOG(INFO) << "#iter: " << accum_iter_ << " #sample: " << accum_sample_
              << " #loss: " << accum_loss_
              << " AVE loss: " << 1.0 * accum_loss_ / accum_iter_;
    accum_iter_ = 0;
    accum_sample_ = 0;
    accum_loss_ = 0;
  }
}

template <typename Dtype>
void YASoftmaxWithLossLayer<Dtype>::Backward_cpu(
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

#ifdef CPU_ONLY
STUB_GPU(YASoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(YASoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(YASoftmaxWithLoss);

}  // namespace caffe

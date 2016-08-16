#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/log_sum_exp_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LogSumExpLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  LogSumExpParameter this_layer_param = this->layer_param_.log_sum_exp_param();
  r_ = this_layer_param.r();
  debug_info_ = false;
  //debug_info_ = true;
  LOG(INFO) << "----------------------------------------------";
  LOG(INFO) << "r_: " << r_;
  LOG(INFO) << "debug_info_: " << debug_info_;
  LOG(INFO) << "----------------------------------------------";
}

template <typename Dtype>
void LogSumExpLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  // current only support ONE im per batch
  num_ = bottom[0]->shape(0);
  channels_ = bottom[0]->shape(1);
  count_ = bottom[0]->count();
  inner_count_ = count_ / num_ / channels_;

  top[0]->Reshape(1, channels_, 1, 1);

  exp_.ReshapeLike(*bottom[0]);
  sum_exp_.Reshape(1, channels_, 1, 1);
  // r_ = log(num_roi);
}

template <typename Dtype>
void LogSumExpLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void LogSumExpLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(LogSumExpLayer);
#endif

INSTANTIATE_CLASS(LogSumExpLayer);
REGISTER_LAYER_CLASS(LogSumExp);

}  // namespace caffe

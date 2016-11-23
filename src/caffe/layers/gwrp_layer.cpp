#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/gwrp_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GWRPLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
  GWRPParameter this_layer_param = this->layer_param_.gwrp_param();
  d_ = this_layer_param.d();
  debug_info_ = false;
  debug_info_ = true;
  LOG(INFO) << "----------------------------------------------";
  LOG(INFO) << "d_: " << d_;
  LOG(INFO) << "debug_info_: " << debug_info_;
  LOG(INFO) << "----------------------------------------------";

  // init weight and sum of weight
  max_num_roi_ = 2048;
  vector<int> weight_shape;
  weight_shape.push_back(max_num_roi_);
  weight_.Reshape(weight_shape);
  weight_sum_.Reshape(weight_shape);

  Dtype* weight_data = weight_.mutable_cpu_data();
  Dtype* weight_sum_data = weight_sum_.mutable_cpu_data();
  weight_data[0] = 1.0;
  weight_sum_data[0] = 1.0;
  for (int n = 1; n < max_num_roi_; ++n) {
    weight_data[n] = weight_data[n - 1] * d_;
    weight_sum_data[n] = weight_sum_data[n - 1] + weight_data[n];
    if (debug_info_)
      std::cout << weight_data[n] << " " << weight_sum_data[n] << std::endl;
  }
}

template <typename Dtype>
void GWRPLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top) {
  // current only support ONE im per batch
  num_roi_ = bottom[0]->shape(0);
  num_class_ = bottom[0]->shape(1);
  count_ = bottom[0]->count();
  normalization_ = weight_sum_.cpu_data()[num_roi_ - 1];
  CHECK_EQ(num_roi_ * num_class_, count_) << "height and width should be 1.";
  CHECK_LE(num_roi_, max_num_roi_) << "num_roi_ should <= max_num_roi_";

  top[0]->Reshape(1, num_class_, 1, 1);
  rank_id_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void GWRPLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void GWRPLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                    const vector<bool>& propagate_down,
                                    const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(GWRPLayer);
#endif

INSTANTIATE_CLASS(GWRPLayer);
REGISTER_LAYER_CLASS(GWRP);

}  // namespace caffe

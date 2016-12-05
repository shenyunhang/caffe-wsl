#include <vector>

#include "caffe/layers/mil_ms_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/opg_layer.hpp"

namespace caffe {

template <typename Dtype>
void MIL_MSLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {}

template <typename Dtype>
void MIL_MSLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top) {
  num_class_ = bottom[0]->channels();
  num_roi_ = bottom[0]->num();
  num_spatial_ = num_roi_ * num_class_;

  vector<int> filter_dims;
  filter_dims.push_back(num_class_);
  filter_dims.push_back(num_roi_);
  filter_dims.push_back(num_class_);
  filter_.Reshape(filter_dims);

  // shape top blob
  top[0]->ReshapeLike(filter_);

  //vector<int> top1_dims;
  //top1_dims.push_back(num_class_);
  //top[1]->Reshape(top1_dims);
  //Dtype* top1_data = top[1]->mutable_cpu_data();
  //for (int c = 0; c < num_class_; ++c) {
    //top1_data[c] = c;
  //}
}

template <typename Dtype>
void MIL_MSLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void MIL_MSLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(MIL_MSLayer);
#endif

INSTANTIATE_CLASS(MIL_MSLayer);
REGISTER_LAYER_CLASS(MIL_MS);

}  // namespace caffe

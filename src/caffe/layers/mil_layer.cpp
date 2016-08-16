#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/mil_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MILLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top) {

  bottom_label_index_ = 0;
  bottom_predict_index_ = 1;
  bottom_rois_index_ = 2;
  bottom_filt_index_ = 3;
  bottom_io_index_ = 4;

  top_select_index_ = 0;
  top_poslabel_index_ = 0;
  top_neglabel_index_ = 0;

  cpg_blob_ = new Blob<Dtype>();

  LayerParameter mil_param(this->layer_param_);

  cpg_layer_ = LayerRegistry<Dtype>::CreateLayer(mil_param);
  cpg_bottom_vec_.clear();
  cpg_bottom_vec_.push_back(bottom[bottom_label_index_]);
  cpg_bottom_vec_.push_back(bottom[bottom_predict_index_]);
  cpg_top_vec_.clear();
  cpg_top_vec_.push_back(cpg_blob_);
  cpg_layer_->SetUp(cpg_bottom_vec_, cpg_top_vec_);

  repartition_layer_ = LayerRegistry<Dtype>::CreateLayer(mil_param);
  repartition_bottom_vec_.clear();
  repartition_bottom_vec_.push_back(cpg_blob_);
  repartition_bottom_vec_.push_back(bottom[bottom_rois_index_]);
  repartition_bottom_vec_.push_back(bottom[bottom_label_index_]);
  repartition_bottom_vec_.push_back(bottom[bottom_predict_index_]);
  repartition_bottom_vec_.push_back(bottom[bottom_filt_index_]);
  repartition_bottom_vec_.push_back(bottom[bottom_io_index_]);
  repartition_top_vec_.clear();
  repartition_top_vec_.push_back(top[0]);
  repartition_layer_->SetUp(repartition_bottom_vec_, repartition_top_vec_);
}

template <typename Dtype>
void MILLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top) {
  cpg_layer_->Reshape(cpg_bottom_vec_, cpg_top_vec_);
  repartition_layer_->Reshape(repartition_bottom_vec_, repartition_bottom_vec_);
}

template <typename Dtype>
void MILLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void MILLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                   const vector<bool>& propagate_down,
                                   const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(MILLayer);
#endif

INSTANTIATE_CLASS(MILLayer);
REGISTER_LAYER_CLASS(MIL);

}  // namespace caffe

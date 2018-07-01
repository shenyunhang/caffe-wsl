#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/bbox_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BBoxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
  BBoxParameter this_layer_param = this->layer_param_.bbox_param();
  is_cpg_ = this_layer_param.is_cpg();
  debug_info_ = this_layer_param.debug_info();
  predict_threshold_ = this_layer_param.predict_threshold();
  crf_threshold_ = this_layer_param.crf_threshold();
  fg_threshold_ = this_layer_param.fg_threshold();
  bg_threshold_ = this_layer_param.bg_threshold();
  mass_threshold_ = this_layer_param.mass_threshold();
  density_threshold_ = this_layer_param.density_threshold();
  is_crf_ = this_layer_param.is_crf();
  is_pred_ = this_layer_param.is_pred();

  bottom_cpgs_index_ = 0;
  bottom_predict_index_ = 1;

  total_im_ = 0;
  total_roi_ = 0;
  total_label_ = 0;

  accum_im_ = 0;
  accum_roi_ = 0;
  accum_label_ = 0;

  max_bb_per_im_ = 4;
  max_bb_per_cls_ = 4;

  LOG(INFO) << "----------------------------------------------";
  LOG(INFO) << "is_cpg: " << is_cpg_;
  LOG(INFO) << "is_pred: " << is_pred_;
  LOG(INFO) << "debug_info_: " << debug_info_;
  LOG(INFO) << "predict_threshold_:" << predict_threshold_;
  LOG(INFO) << "fg_threshold_:" << fg_threshold_;
  LOG(INFO) << "bg_threshold_:" << bg_threshold_;
  LOG(INFO) << "mass_threshold_:" << mass_threshold_;
  LOG(INFO) << "density_threshold_:" << density_threshold_;
  LOG(INFO) << "max_bb_per_im_:" << max_bb_per_im_;
  LOG(INFO) << "max_bb_per_cls_:" << max_bb_per_cls_;
  LOG(INFO) << "is_crf_:" << is_crf_;
  LOG(INFO) << "----------------------------------------------";

  if (is_crf_) {
    crf_bottom_vec_.clear();
    crf_bottom_vec_.push_back(crf_cpg_.get());
    crf_bottom_vec_.push_back(crf_data_dim_.get());
    crf_bottom_vec_.push_back(crf_data_.get());
    crf_top_vec_.clear();
    crf_top_vec_.push_back(crf_output_.get());

    crf_data_->Reshape(1, 3, 1000, 1000);
    crf_cpg_->Reshape(1, 1, 1000, 1000);
    crf_data_dim_->Reshape(1, 2, 1, 1);
    crf_layer_->SetUp(crf_bottom_vec_, crf_top_vec_);
  }

  if (debug_info_) {
    voc_label_.push_back("aeroplane");
    voc_label_.push_back("bicycle");
    voc_label_.push_back("bird");
    voc_label_.push_back("boat");
    voc_label_.push_back("bottle");
    voc_label_.push_back("bus");
    voc_label_.push_back("car");
    voc_label_.push_back("cat");
    voc_label_.push_back("chair");
    voc_label_.push_back("cow");
    voc_label_.push_back("diningtable");
    voc_label_.push_back("dog");
    voc_label_.push_back("horse");
    voc_label_.push_back("motorbike");
    voc_label_.push_back("person");
    voc_label_.push_back("pottedplant");
    voc_label_.push_back("sheep");
    voc_label_.push_back("sofa");
    voc_label_.push_back("train");
    voc_label_.push_back("tvmonitor");
  }
}

template <typename Dtype>
void BBoxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top) {
  num_class_ = bottom[bottom_predict_index_]->channels();
  num_im_ = bottom[bottom_predict_index_]->num();
  CHECK_EQ(num_im_, 1) << "current only support one image per forward-backward";

  // shape top
  vector<int> top_shape;
  top_shape.push_back(num_class_);
  top_shape.push_back(max_bb_per_cls_);
  top_shape.push_back(4);
  top[0]->Reshape(top_shape);
  caffe_set(top[0]->count(), Dtype(-1), top[0]->mutable_cpu_data());

  // shape bboxes_
  vector<int> bboxes_shape;
  bboxes_shape.push_back(max_bb_per_cls_);
  bboxes_shape.push_back(4);
  bboxes_->Reshape(bboxes_shape);

  channels_cpg_ = bottom[bottom_cpgs_index_]->channels();
  height_im_ = bottom[bottom_cpgs_index_]->height();
  width_im_ = bottom[bottom_cpgs_index_]->width();
  cpg_size_ = height_im_ * width_im_;

  raw_cpg_->Reshape(1, 1, height_im_, width_im_);

  CHECK_EQ(bottom[bottom_predict_index_]->count(), num_im_ * num_class_)
      << "size should be the same";
}

template <typename Dtype>
void BBoxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void BBoxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                    const vector<bool>& propagate_down,
                                    const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(BBoxLayer);
#endif

INSTANTIATE_CLASS(BBoxLayer);
REGISTER_LAYER_CLASS(BBox);

}  // namespace caffe

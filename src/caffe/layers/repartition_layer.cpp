#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/repartition_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RepartitionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  CPGParameter this_layer_param = this->layer_param_.cpg_param();
  is_opg_ = this_layer_param.is_cpg();
  debug_info_ = this_layer_param.debug_info();
  is_crf_ = this_layer_param.is_crf();
  is_pred_ = this_layer_param.is_pred();
  is_order_ = this_layer_param.is_order();
  is_softmax_ = this_layer_param.is_softmax();

  ignore_label_ = this_layer_param.ignore_label();

  predict_threshold_ = this_layer_param.predict_threshold();
  predict_order_ = this_layer_param.predict_order();
  crf_threshold_ = this_layer_param.crf_threshold();
  fg_threshold_ = this_layer_param.fg_threshold();
  bg_threshold_ = this_layer_param.bg_threshold();
  mass_threshold_ = this_layer_param.mass_threshold();
  density_threshold_ = this_layer_param.density_threshold();

  bottom_opgs_index_ = 0;
  bottom_rois_index_ = 1;
  bottom_label_index_ = 2;
  bottom_predict_index_ = 3;
  bottom_filt_index_ = 4;
  bottom_io_index_ = 5;

  total_im_ = 0;
  total_roi_ = 0;
  total_roi_l_ = 0;
  total_label_ = 0;

  accum_im_ = 0;
  accum_roi_ = 0;
  accum_roi_l_ = 0;
  accum_label_ = 0;

  max_bb_per_im_ = 4;
  max_bb_per_cls_ = 4;

  if (is_order_) {
    order_K_ = 3;
    order_step_ = 10022 * 3;
    order_threshold_ = 1.0 * (order_K_ - 1) / order_K_;
    CHECK_EQ(top.size(), 3) << "In size order mode, #top should be 3!";
  }

  LOG(INFO) << "----------------------------------------------";
  LOG(INFO) << "is_opg_: " << is_opg_;
  LOG(INFO) << "is_order_: " << is_order_;
  LOG(INFO) << "is_pred_: " << is_pred_;
  LOG(INFO) << "debug_info_: " << debug_info_;
  LOG(INFO) << "predict_threshold_:" << predict_threshold_;
  LOG(INFO) << "predict_order_:" << predict_order_;
  LOG(INFO) << "fg_threshold_:" << fg_threshold_;
  LOG(INFO) << "bg_threshold_:" << bg_threshold_;
  LOG(INFO) << "mass_threshold_:" << mass_threshold_;
  LOG(INFO) << "density_threshold_:" << density_threshold_;
  LOG(INFO) << "max_bb_per_cls_:" << max_bb_per_cls_;
  LOG(INFO) << "is_crf_:" << is_crf_;
  LOG(INFO) << "----------------------------------------------";

  if (is_crf_) {
    crf_bottom_vec_.clear();
    crf_bottom_vec_.push_back(crf_opg_.get());
    crf_bottom_vec_.push_back(crf_data_dim_.get());
    crf_bottom_vec_.push_back(crf_data_.get());
    crf_top_vec_.clear();
    crf_top_vec_.push_back(crf_output_.get());

    crf_data_->Reshape(1, 3, 1000, 1000);
    crf_opg_->Reshape(1, 1, 1000, 1000);
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
    voc_label_.push_back("background");
  }
}

template <typename Dtype>
void RepartitionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  num_roi_ = bottom[bottom_rois_index_]->num();
  num_class_ = bottom[bottom_label_index_]->channels();
  num_im_ = bottom[bottom_label_index_]->num();
  CHECK_EQ(num_im_, 1) << "current only support one image per forward-backward";

  // shape fliter
  vector<int> fliter_shape;
  fliter_shape.push_back(num_roi_);
  fliter_shape.push_back(num_class_);
  fliter_.Reshape(fliter_shape);

  // shape top
  if (is_softmax_) {
    vector<int> top_shape;
    top_shape.push_back(num_roi_);
    top[0]->Reshape(top_shape);
  } else {
    vector<int> top_shape;
    top_shape.push_back(num_roi_);
    top_shape.push_back(num_class_);
    top[0]->Reshape(top_shape);
  }

  if (top.size() == 3) {
    top[1]->CopyFrom(*bottom[bottom_label_index_], false, true);
    top[2]->ReshapeLike(*bottom[bottom_label_index_]);
    caffe_set(top[2]->count(), Dtype(0), top[2]->mutable_cpu_data());
  }
  // shape bboxes_
  vector<int> bboxes_shape;
  bboxes_shape.push_back(max_bb_per_cls_);
  bboxes_shape.push_back(4);
  bboxes_->Reshape(bboxes_shape);

  channels_opg_ = bottom[bottom_opgs_index_]->channels();
  height_im_ = bottom[bottom_opgs_index_]->height();
  width_im_ = bottom[bottom_opgs_index_]->width();
  opg_size_ = height_im_ * width_im_;

  raw_opg_->Reshape(1, 1, height_im_, width_im_);

  CHECK_EQ(bottom[bottom_predict_index_]->shape(0), num_im_)
      << "#im should be the same";
  CHECK_EQ(bottom[bottom_predict_index_]->shape(1), num_class_)
      << "#class should be the same";
  CHECK_EQ(bottom[bottom_predict_index_]->count(), num_im_ * num_class_)
      << "size should be the same";
  CHECK_EQ(bottom[bottom_label_index_]->count(), num_im_ * num_class_)
      << "size should be the same";

  if (bottom.size() > bottom_io_index_) {
    CHECK_EQ(bottom[bottom_filt_index_]->shape(0), num_roi_)
        << "#roi should be the same";
    CHECK_EQ(bottom[bottom_filt_index_]->shape(1), num_class_)
        << "#class should be the same";
    CHECK_EQ(bottom[bottom_io_index_]->count(), 1) << "only need one IO ID";
  }
}

template <typename Dtype>
void RepartitionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void RepartitionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down,
                                           const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(RepartitionLayer);
#endif

INSTANTIATE_CLASS(RepartitionLayer);
REGISTER_LAYER_CLASS(Repartition);

}  // namespace caffe

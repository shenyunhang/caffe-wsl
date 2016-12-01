#include <vector>

#include "caffe/layers/opg_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void OPGLayer<Dtype>::Get_split_top_blob() {
  const vector<shared_ptr<Layer<Dtype> > > layers = net_->layers();
  const vector<vector<Blob<Dtype>*> > top_vecs = net_->top_vecs();
  const vector<string> blob_names = net_->blob_names();
  const vector<string> layer_names = net_->layer_names();

  LOG(INFO) << "----------------------------------------------";
  LOG(INFO) << "Top blob diff of split layer to clear";
  for (int layer_id = start_layer_index_; layer_id < end_layer_index_;
       ++layer_id) {
    shared_ptr<Layer<Dtype> > this_layer = layers[layer_id];
    const char* type = this_layer->type();
    if (strcmp(type, "Split") == 0) {
      const vector<int> top_ids = net_->top_ids(layer_id);
      const vector<Blob<Dtype>*> top_blobs = top_vecs[layer_id];
      LOG(INFO) << "Layer: " << layer_names[layer_id];
      for (int blob_id = 0; blob_id < top_blobs.size(); ++blob_id) {
        split_top_blob_.push_back(top_blobs[blob_id]);
        LOG(INFO) << "\tblob name: " << blob_names[top_ids[blob_id]];
      }
    }
  }
  LOG(INFO) << "----------------------------------------------";
}

template <typename Dtype>
void OPGLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                 const vector<Blob<Dtype>*>& top) {
  CPGParameter this_layer_param = this->layer_param_.cpg_param();
  LOG(INFO) << "type: " << this->layer_param_.type();
  is_opg_ = this_layer_param.is_cpg();
  is_order_ = this_layer_param.is_order();
  debug_info_ = this_layer_param.debug_info();
  is_contrast_ = this_layer_param.is_contrast();
  ignore_label_ = this_layer_param.ignore_label();
  predict_threshold_ = this_layer_param.predict_threshold();
  predict_order_ = this_layer_param.predict_order();
  start_layer_name_ = this_layer_param.start_layer_name();
  end_layer_name_ = this_layer_param.end_layer_name();
  for (size_t i = 0; i < this_layer_param.opg_blob_name().size(); ++i) {
    opg_blob_name_.push_back(this_layer_param.opg_blob_name(i));
  }

  bottom_label_index_ = 0;
  bottom_predict_index_ = 1;

  save_id_ = 0;
  is_show_ = true;

  history_params_.clear();
  history_blobs_.clear();
  is_history_init_ = false;

  // init layer index
  const vector<string> layer_names = net_->layer_names();
  for (size_t i = 0; i < layer_names.size(); i++) {
    if (layer_names[i].compare(start_layer_name_) == 0) {
      start_layer_index_ = i;
    }
    if (layer_names[i].compare(end_layer_name_) == 0) {
      end_layer_index_ = i;
    }
  }

  opg_blob_index_.clear();
  const vector<string> blob_names = net_->blob_names();
  for (size_t i = 0; i < blob_names.size(); i++) {
    for (size_t j = 0; j < opg_blob_name_.size(); ++j) {
      if (blob_names[i].compare(opg_blob_name_[j]) == 0) {
        opg_blob_index_.push_back(i);
      }
    }
  }

  CHECK_EQ(opg_blob_name_.size(), opg_blob_index_.size())
      << "some opg layers can not find!";

  CHECK_LT(start_layer_index_, end_layer_index_)
      << "OPGLayer: start_layer_index_ should below end_layer_index_";

  if (opg_blob_name_.size() == 0) {
    is_opg_ = false;
    LOG(INFO) << "Change is_opg to false, due to no opg_blob_name.";
  }

  // find net blob
  {
    LOG(INFO) << "finding net blob";
    // set im blob and predictiob blob
    const vector<int> start_bottom_ids = net_->bottom_ids(start_layer_index_);
    CHECK_EQ(start_bottom_ids.size(), 1)
        << "start_bottom_ids size should be one";
    image_blob_index_ = start_bottom_ids[0];
    im_blob_ = net_->blobs()[image_blob_index_];

    const vector<int> end_top_ids = net_->top_ids(end_layer_index_);
    CHECK_EQ(end_top_ids.size(), 1) << "end_top_ids size should be one";
    predict_blob_index_ = end_top_ids[0];
    predict_blob_ = net_->blobs()[predict_blob_index_];

    CHECK_LT(start_bottom_ids[0], end_top_ids[0])
        << "start_bottom_ids should be small than end_top_ids";

    for (size_t i = 0; i < opg_blob_index_.size(); ++i) {
      // const vector<int> opg_top_ids = net_->top_ids(opg_layer_index_[i]);
      // CHECK_EQ(opg_top_ids.size(), 1) << "opg_top_ids size should be one";
      // opg_blob_.push_back(net_->blobs()[opg_top_ids[0]]);
      opg_blob_.push_back(net_->blobs()[opg_blob_index_[i]]);
    }
  }

  Get_split_top_blob();

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

  LOG(INFO) << "==============================================";
  LOG(INFO) << "OPG layer:";
  LOG(INFO) << "is_opg_: " << is_opg_;
  LOG(INFO) << "ignore_label_: " << ignore_label_;
  LOG(INFO) << "debug_info_: " << debug_info_;
  LOG(INFO) << "is_contrast: " << is_contrast_;
  LOG(INFO) << "predict_threshold_:" << predict_threshold_;
  LOG(INFO) << "start_layer_name_: " << start_layer_name_;
  LOG(INFO) << "start_layer_index_: " << start_layer_index_;
  LOG(INFO) << "end_layer_name_: " << end_layer_name_;
  LOG(INFO) << "end_layer_index_: " << end_layer_index_;
  LOG(INFO) << "image_blob_index_: " << image_blob_index_;
  LOG(INFO) << "predict_blob_index_: " << predict_blob_index_;
  for (size_t i = 0; i < opg_blob_name_.size(); ++i) {
    LOG(INFO) << "opg_blob_name_: " << opg_blob_name_[i];
    LOG(INFO) << "opg_blob_index_: " << opg_blob_index_[i];
  }
  LOG(INFO) << "==============================================";
}

template <typename Dtype>
void OPGLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                              const vector<Blob<Dtype>*>& top) {
  num_class_ = bottom[bottom_label_index_]->channels();
  num_im_ = bottom[bottom_label_index_]->num();
  CHECK_EQ(num_im_, 1) << "current only support one image per forward-backward";

  channels_im_ = im_blob_->channels();
  height_im_ = im_blob_->height();
  width_im_ = im_blob_->width();
  channels_opg_ = 1;
  opg_size_ = height_im_ * width_im_;

  // tmp shape top
  vector<int> top_shape;
  top_shape.push_back(num_im_);
  top_shape.push_back(1);
  top_shape.push_back(height_im_);
  top_shape.push_back(width_im_);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void OPGLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void OPGLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                   const vector<bool>& propagate_down,
                                   const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void OPGLayer<Dtype>::After() {
  accum_im_ += num_im_;
  accum_gt_ += gt_class_.size();
  accum_bp_ += bp_class_.size();
  if (accum_im_ == 1280) {
    LOG(INFO) << "#im: " << accum_im_ << " #bp: " << accum_bp_
              << " #gt: " << accum_gt_;
    accum_im_ = 0;
    accum_gt_ = 0;
    accum_bp_ = 0;
  }
  save_id_ += num_im_;
}

#ifdef CPU_ONLY
STUB_GPU(OPGLayer);
#endif

INSTANTIATE_CLASS(OPGLayer);
REGISTER_LAYER_CLASS(OPG);

}  // namespace caffe

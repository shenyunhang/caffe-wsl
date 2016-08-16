// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/layers/loop_roi_pooling_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void LoopROIPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  LoopROIPoolingParameter loop_roi_pool_param =
      this->layer_param_.loop_roi_pooling_param();
  CHECK_GT(loop_roi_pool_param.pooled_h(), 0) << "pooled_h must be > 0";
  CHECK_GT(loop_roi_pool_param.pooled_w(), 0) << "pooled_w must be > 0";
  pooled_height_ = loop_roi_pool_param.pooled_h();
  pooled_width_ = loop_roi_pool_param.pooled_w();
  spatial_scale_ = loop_roi_pool_param.spatial_scale();
  LOG(INFO) << "Spatial scale: " << spatial_scale_;
}

template <typename Dtype>
void LoopROIPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_);
  max_idx_.Reshape(bottom[1]->num(), channels_, pooled_height_, pooled_width_);
}

template <typename Dtype>
void LoopROIPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void LoopROIPoolingLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(LoopROIPoolingLayer);
#endif

INSTANTIATE_CLASS(LoopROIPoolingLayer);
REGISTER_LAYER_CLASS(LoopROIPooling);

}  // namespace caffe

#ifndef CAFFE_SMOOTH_FEATURE_LOSS_LAYER_HPP_
#define CAFFE_SMOOTH_FEATURE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class SmoothFeatureLossLayer : public LossLayer<Dtype> {
 public:
  explicit SmoothFeatureLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SmoothFeatureLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 4; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  // virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  // const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  // virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
  // const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Dtype min_overlap_;
  int max_num_bbox_;
  Dtype threshold_;
  bool debug_info_;
  bool is_sigmoid_;

  Blob<int> max_bbox_idx_;
  Blob<int> bbox_idx_;
  Blob<Dtype> bbox_prob_;
  Blob<Dtype> diff_;

  Blob<int>* indices_blob_;
  int total_num_this_;
  int total_num_;
  int total_im_;
  Dtype total_loss_;

  int feature_dim_;
  int num_roi_;
  int num_im_;
  int num_class_;
};

}  // namespace caffe

#endif  // CAFFE_SMOOTH_FEATURE_LAYER_HPP_

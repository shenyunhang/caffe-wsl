#ifndef CAFFE_YA_SOFTMAX_WITH_LOSS_LAYER_HPP_
#define CAFFE_YA_SOFTMAX_WITH_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"

namespace caffe {

template <typename Dtype>
class YASoftmaxWithLossLayer : public LossLayer<Dtype> {
 public:
  explicit YASoftmaxWithLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "YASoftmaxWithLoss"; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  // The internal SoftmaxWithLossLayer.
  shared_ptr<Layer<Dtype> > softmax_loss_layer_;
  // bottom vector holder used in call to the underlying SoftmaxWithLossLayer::Forward
  vector<Blob<Dtype>*> softmax_loss_bottom_vec_;
  // top vector holder used in call to the underlying SoftmaxWithLossLayer::Forward
  vector<Blob<Dtype>*> softmax_loss_top_vec_;
  // Whether to ignore instances with a certain label.
  bool has_ignore_label_;
  // The label indicating that an instance should be ignored.
  int ignore_label_;

  int total_iter_;
  int total_sample_;
  Dtype total_loss_;

  int accum_iter_;
  int accum_sample_;
  Dtype accum_loss_;
};

}  // namespace caffe

#endif  // CAFFE_YA_SOFTMAX_WITH_LOSS_LAYER_HPP_

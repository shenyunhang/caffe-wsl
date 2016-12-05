#ifndef CAFFE_WEIGHTED_CENTER_LOSS_LAYER_HPP_
#define CAFFE_WEIGHTED_CENTER_LOSS_LAYER_HPP_

#include <vector>
#include <iostream>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class WeightedCenterLossLayer : public LossLayer<Dtype> {
 public:
  explicit WeightedCenterLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "WeightedCenterLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  // bottom:
  // 1: feature
  // 2: label
  // 3: bbox score

  Blob<Dtype> diff_;
  // We repalce the center_ with blob_.
  // So that the center will be saved when the net is saved.
  // However we don't update the center by the net, but by this layer itself.
  // So the learning rate of this layer is set to zero.
  // Blob<Dtype> center_;
  int num_center_;

  bool debug_info_;
  Dtype lr_;
  int dim_;
  int num_class_;

  int num_gt_class_;
  int num_roi_;

  vector<int> center_selector_;
  vector<Dtype> loss_scale_;

  Dtype accum_loss_;
  int total_iter_;
  int display_;

  int update_;
  // for update centers
  vector<vector<int> > num_update_class_;
  // for display
  vector<vector<int> > accum_update_class_;
};

}  // namespace caffe

#endif  // CAFFE_WEIGHTED_CENTER_LOSS_LAYER_HPP_

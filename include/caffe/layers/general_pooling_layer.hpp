#ifndef CAFFE_GENERAL_POOLING_LAYER_HPP_
#define CAFFE_GENERAL_POOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class GeneralPoolingLayer : public Layer<Dtype> {
 public:
  explicit GeneralPoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "GeneralPooling"; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int MinBottomBlobs() const { return 1; }
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

  Blob<int> mask_idx_;
  Dtype threshold_;

  int outer_num_;
  int inner_num_;
  int channels_;
  int dim_;

  // TODO(YH): Not all pooling modes  support axis parameter.
  int pooling_axis_;
};

}  // namespace caffe

#endif  // CAFFE_GENERAL_POOLING_LAYER_HPP_

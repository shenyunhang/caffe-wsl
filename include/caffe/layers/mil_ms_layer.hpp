#ifndef CAFFE_MIL_MS_MS_LAYER_HPP_
#define CAFFE_MIL_MS_MS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief
 *
 * TODO(YH):
 */
template <typename Dtype>
class MIL_MSLayer : public Layer<Dtype> {
 public:
  explicit MIL_MSLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MIL_MS"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  // virtual inline int MaxNumBottomBlobs() const { return 5; }
  // virtual inline int MinNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  // virtual inline int MaxNumTopBlobs() const { return 3; }
  // virtual inline int MinNumTopBlobs() const { return 1; }

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

  int num_roi_;
  int num_class_;
  int num_spatial_;

  Blob<Dtype> filter_;
};

}  // namespace caffe

#endif  // CAFFE_MIL_MS_MS_LAYER_HPP_

#ifndef CAFFE_MIL_LAYER_HPP_
#define CAFFE_MIL_LAYER_HPP_

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
class MILLayer : public Layer<Dtype> {
 public:
  explicit MILLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MIL"; }
  //virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MaxNumBottomBlobs() const { return 5; }
  virtual inline int MinNumBottomBlobs() const { return 3; }
  //virtual inline int ExactNumTopBlobs() const { return 1; }
  virtual inline int MaxNumTopBlobs() const { return 3; }
  virtual inline int MinNumTopBlobs() const { return 1; }

  void Set_Net(Net<Dtype>* net) {
    LOG(INFO) << "Setting net in OPGLayer";
    net_ = net;
  }

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

  Net<Dtype>* net_;

  int bottom_label_index_;
  int bottom_predict_index_;
  int bottom_rois_index_;
  int bottom_filter_index_;
  int bottom_io_index_;

  int top_select_index_;
  int top_poslabel_index_;
  int top_neglabel_index_;

  shared_ptr<Layer<Dtype> > cpg_layer_;
  vector<Blob<Dtype>*> cpg_bottom_vec_;
  vector<Blob<Dtype>*> cpg_top_vec_;

  shared_ptr<Layer<Dtype> > repartition_layer_;
  vector<Blob<Dtype>*> repartition_bottom_vec_;
  vector<Blob<Dtype>*> repartition_top_vec_;

  Blob<Dtype> cpg_blob_;
};

}  // namespace caffe

#endif  // CAFFE_MIL_LAYER_HPP_

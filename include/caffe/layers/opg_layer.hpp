#ifndef CAFFE_OPG_LAYER_HPP_
#define CAFFE_OPG_LAYER_HPP_

#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

const float kLOG_THRESHOLD = 1e-20;

template <typename Dtype>
class OPGLayer : public Layer<Dtype> {
 public:
  explicit OPGLayer(const LayerParameter& param)
      : Layer<Dtype>(param), raw_opg_(new Blob<Dtype>()) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OPG"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  void Set_Net(Net<Dtype>* net) {
    LOG(INFO) << "Setting net in OPGLayer";
    net_ = net;
  }

  void OPG_back();

  void Save_param_diff();
  void Restore_param_diff();
  void Show_info();
  void BackwardDebugInfo(const int layer_id);
  void Clear_split_diff();
  void Get_split_top_blob();
  void Show_opg(const Dtype* opg_data, const int cur, const string info = "");
  bool Need_Repartition(const int cls_id, const Dtype label, const Dtype score);
  bool Need_Order(const Dtype label, const Dtype score);

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

  bool is_opg_;
  bool debug_info_;
  bool is_contrast_;
  bool is_order_;
  float predict_threshold_;
  float predict_order_;
  string start_layer_name_;
  string end_layer_name_;
  vector<string> opg_blob_name_;

  int ignore_label_;

  int start_layer_index_;
  int end_layer_index_;
  int image_blob_index_;
  int predict_blob_index_;
  vector<int> opg_blob_index_;

  shared_ptr<Blob<Dtype> > im_blob_;
  shared_ptr<Blob<Dtype> > predict_blob_;
  vector<shared_ptr<Blob<Dtype> > > opg_blob_;

  int bottom_label_index_;
  int bottom_predict_index_;

  int save_id_;
  bool is_show_;
  shared_ptr<Blob<Dtype> > raw_opg_;

  // abandon now
  vector<shared_ptr<Blob<Dtype> > > history_params_;
  vector<shared_ptr<Blob<Dtype> > > history_blobs_;
  bool is_history_init_;

  bool net_blob_init_;
  vector<Blob<Dtype>*> split_top_blob_;
  vector<vector<bool> > bottom_need_backward_;
  vector<vector<bool> > origin_param_propagate_down_;

  int num_class_;
  int num_im_;
  int height_im_;
  int width_im_;
  int channels_im_;
  int channels_opg_;
  int opg_size_;

  vector<string> voc_label_;
};

}  // namespace caffe

#endif  // CAFFE_OPG_LAYER_HPP_

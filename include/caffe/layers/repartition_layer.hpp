#ifndef CAFFE_REPARTITION_LAYER_HPP_
#define CAFFE_REPARTITION_LAYER_HPP_

#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/densecrf_layer.hpp"

namespace caffe {

const float kLOG_THRESHOLD = 1e-20;

template <typename Dtype>
class RepartitionLayer : public Layer<Dtype> {
 public:
  explicit RepartitionLayer(const LayerParameter& param)
      : Layer<Dtype>(param),
        raw_opg_(new Blob<Dtype>()),
        bboxes_(new Blob<Dtype>()) {}
  // explicit RepartitionLayer(const LayerParameter& param)
  //: Layer<Dtype>(param),
  // crf_layer_(new DenseCRFLayer<Dtype>(param)),
  // crf_output_(new Blob<Dtype>()),
  // crf_opg_(new Blob<Dtype>()),
  // crf_data_dim_(new Blob<Dtype>()),
  // crf_data_(new Blob<Dtype>()) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Repartition"; }
  virtual inline int MaxNumBottomBlobs() const { return 6; }
  virtual inline int MinNumBottomBlobs() const { return 4; }
  virtual inline int MaxNumTopBlobs() const { return 3; }
  virtual inline int MinNumTopBlobs() const { return 1; }

  bool aou_small(const Dtype* roi, const Dtype bb_offset);
  void Repartition_crf(const int label);
  void Score_map_crf();
  void InitFilter(const Dtype* const label_gpu_data, Dtype* const top_gpu_data);
  void After();
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

  bool is_opg_;
  bool debug_info_;
  bool is_crf_;
  bool is_pred_;
  bool is_order_;
  bool is_softmax_;

  int ignore_label_;

  float predict_threshold_;
  float predict_order_;
  float crf_threshold_;
  float fg_threshold_;
  float bg_threshold_;
  float mass_threshold_;
  float density_threshold_;

  int bottom_opgs_index_;
  int bottom_rois_index_;
  int bottom_label_index_;
  int bottom_predict_index_;
  int bottom_filt_index_;
  int bottom_io_index_;
  Blob<Dtype>* raw_opg_;
  Blob<Dtype> fliter_;

  int total_im_;
  int total_roi_;
  int total_roi_l_;
  int total_label_;

  int accum_im_;
  int accum_roi_;
  int accum_roi_l_;
  int accum_label_;

  int order_K_;
  float order_threshold_;
  int order_step_;

  Blob<Dtype>* bboxes_;
  int max_bb_per_im_;
  int max_bb_per_cls_;

  int num_class_;
  int num_im_;
  int num_roi_;
  int height_im_;
  int width_im_;
  int channels_opg_;
  int opg_size_;

  // crf
  shared_ptr<DenseCRFLayer<Dtype> > crf_layer_;
  shared_ptr<Blob<Dtype> > crf_output_;
  shared_ptr<Blob<Dtype> > crf_opg_;
  shared_ptr<Blob<Dtype> > crf_data_dim_;
  shared_ptr<Blob<Dtype> > crf_data_;
  vector<Blob<Dtype>*> crf_bottom_vec_;
  vector<Blob<Dtype>*> crf_top_vec_;

  vector<string> voc_label_;
};

}  // namespace caffe

#endif  // CAFFE_REPARTITION_LAYER_HPP_

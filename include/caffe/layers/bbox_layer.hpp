#ifndef CAFFE_BBOX_LAYER_HPP_
#define CAFFE_BBOX_LAYER_HPP_

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
class BBoxLayer : public Layer<Dtype> {
 public:
  explicit BBoxLayer(const LayerParameter& param)
      : Layer<Dtype>(param),
        raw_cpg_(new Blob<Dtype>()),
        bboxes_(new Blob<Dtype>()) {}
  // explicit BBoxLayer(const LayerParameter& param)
  //: Layer<Dtype>(param),
  // raw_cpg_(new Blob<Dtype>()),
  // crf_layer_(new DenseCRFLayer<Dtype>(param)),
  // crf_output_(new Blob<Dtype>()),
  // crf_cpg_(new Blob<Dtype>()),
  // crf_data_dim_(new Blob<Dtype>()),
  // crf_data_(new Blob<Dtype>()) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BBox"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  bool aou_small(const Dtype* roi, const Dtype bb_offset);
  void BBox_crf(const int label);
  void Score_map_crf();
  bool Need_Back(const Dtype label, const Dtype score);
  void After();

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

  bool is_cpg_;
  bool debug_info_;
  float predict_threshold_;
  float crf_threshold_;
  float fg_threshold_;
  float bg_threshold_;
  float mass_threshold_;
  float density_threshold_;
  bool is_crf_;
  bool is_pred_;

  int bottom_cpgs_index_;
  int bottom_predict_index_;
  Blob<Dtype>* raw_cpg_;

  int total_im_;
  int total_roi_;
  int total_label_;

  int accum_im_;
  int accum_roi_;
  int accum_label_;

  Blob<Dtype>* bboxes_;
  int max_bb_per_im_;
  int max_bb_per_cls_;

  int num_class_;
  int num_im_;
  int height_im_;
  int width_im_;
  int channels_cpg_;
  int cpg_size_;

  // crf
  shared_ptr<DenseCRFLayer<Dtype> > crf_layer_;
  shared_ptr<Blob<Dtype> > crf_output_;
  shared_ptr<Blob<Dtype> > crf_cpg_;
  shared_ptr<Blob<Dtype> > crf_data_dim_;
  shared_ptr<Blob<Dtype> > crf_data_;
  vector<Blob<Dtype>*> crf_bottom_vec_;
  vector<Blob<Dtype>*> crf_top_vec_;

  vector<string> voc_label_;
};

}  // namespace caffe

#endif  // CAFFE_BBOX_LAYER_HPP_

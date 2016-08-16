#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/smooth_feature_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 1. / (1. + exp(-x));
}

template <typename Dtype>
void SmoothFeatureLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  max_num_bbox_ = this->layer_param_.smooth_feature_loss_param().max_num_bbox();
  min_overlap_ = this->layer_param_.smooth_feature_loss_param().min_overlap();
  threshold_ = this->layer_param_.smooth_feature_loss_param().threshold();
  debug_info_ = this->layer_param_.smooth_feature_loss_param().debug_info();
  is_sigmoid_ = this->layer_param_.smooth_feature_loss_param().is_sigmoid();
  LOG(INFO) << "----------------------------------------------";
  LOG(INFO) << "max_num_bbox_: " << max_num_bbox_;
  LOG(INFO) << "min_overlap_: " << min_overlap_;
  LOG(INFO) << "threshold_: " << threshold_;
  LOG(INFO) << "----------------------------------------------";
  indices_blob_ = new Blob<int>(1, 1, 1, 1);
  total_num_ = 0;
  total_im_ = 0;
  total_loss_ = 0;
}

template <typename Dtype>
void SmoothFeatureLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  // LossLayer<Dtype>::Reshape(bottom, top);
  // 用下面两替代LossLayer<Dtype>::Reshape(bottom, top) 以此跳过里面的CHECK
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);

  // feature		#roi	c	h	w
  // rois		#roi	5	1	1
  // score		#roi	#class	1	1
  // label		#im	#class

  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0));
  CHECK_EQ(bottom[0]->shape(0), bottom[2]->shape(0));
  CHECK_EQ(bottom[2]->shape(1), bottom[3]->shape(1));

  num_roi_ = bottom[0]->shape(0);
  num_im_ = bottom[3]->shape(0);
  num_class_ = bottom[3]->shape(1);
  max_bbox_idx_.Reshape(num_im_, num_class_, 1, 1);
  bbox_idx_.Reshape(num_im_, num_class_, max_num_bbox_, 1);
  bbox_prob_.Reshape(num_im_, num_class_, max_num_bbox_, 1);

  CHECK_EQ(bottom[0]->num_axes(), 2)
      << "current support feature blob has axes 2.";

  feature_dim_ = bottom[0]->count() / bottom[0]->num();
  diff_.Reshape(num_im_, num_class_, max_num_bbox_, feature_dim_);
  indices_blob_->Reshape(num_roi_, 1, 1, 1);

  total_num_this_ = 0;
}

template <typename Dtype>
void ordered(const Dtype* values, int* indices, const int count,
             const int stride, const int top_n) {
  for (int p = 0; p < count; ++p) {
    indices[p] = p;
  }

  for (int i = 0; i < top_n && i < count; ++i) {
    int t_i = indices[i];
    int max_p = i;
    int max_t_p = t_i;
    for (int j = i + 1; j < count; ++j) {
      // int t_j = (indices[j] == -1) ? j : indices[j];
      int t_j = indices[j];

      if (values[max_t_p * stride] < values[t_j * stride]) {
        max_p = j;
        max_t_p = t_j;
      }
    }
    indices[i] = max_t_p;
    indices[max_p] = t_i;
  }
}

template <typename Dtype>
void ordered_at(const Dtype* values, int* indices, const int i, const int count,
                const int stride) {
  int t_i = indices[i];
  int max_p = i;
  int max_t_p = t_i;
  for (int j = i + 1; j < count; ++j) {
    // int t_j = (indices[j] == -1) ? j : indices[j];
    int t_j = indices[j];

    if (values[max_t_p * stride] < values[t_j * stride]) {
      max_p = j;
      max_t_p = t_j;
    }
  }
  indices[i] = max_t_p;
  indices[max_p] = t_i;
}

template <typename Dtype>
void SmoothFeatureLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // feature		#roi	c	h	w
  // rois		#roi	5	1	1
  // score		#roi	#class	1	1
  // label		#im	#class

  if (debug_info_) LOG(INFO) << "SmoothFeatureLossLayer Forward_cpu";

  const Dtype* feature = bottom[0]->cpu_data();
  const Dtype* rois = bottom[1]->cpu_data();
  const Dtype* scores = bottom[2]->cpu_data();
  const Dtype* label = bottom[3]->cpu_data();
  Dtype* loss = top[0]->mutable_cpu_data();

  caffe_set(bbox_idx_.count(), int(-1), bbox_idx_.mutable_cpu_data());
  int* max_bbox_idx = max_bbox_idx_.mutable_cpu_data();
  int* bbox_idx = bbox_idx_.mutable_cpu_data();
  Dtype* bbox_prob = bbox_prob_.mutable_cpu_data();
  Dtype* diff;
  int* indices = indices_blob_->mutable_cpu_data();

  loss[0] = 0;
  for (int n = 0; n < num_im_; ++n) {
    CHECK_EQ(n, 0) << "current only support one image per minibatch";
    for (int c = 0; c < num_class_; ++c) {
      if (label[n * num_class_ + c] == 0) continue;

      // the #roi per im can't be known before
      ordered(scores + c, indices, num_roi_, num_class_, max_num_bbox_);

      int i = indices[0];
      max_bbox_idx[max_bbox_idx_.offset(n, c, 0, 0)] = i;
      Dtype s_i = (rois[i * 5 + 3] - rois[i * 5 + 1] + 1) *
                  (rois[i * 5 + 4] - rois[i * 5 + 2] + 1);
      if (debug_info_)
        LOG(INFO) << "c: " << c << " i: " << i
                  << " scores: " << scores[i * num_class_ + c];
      for (int r = 0; r < max_num_bbox_; ++r) {
        // int roi_batch_ind = bottom_rois[0];
        // int roi_start_w = round(bottom_rois[1] * spatial_scale_);
        // int roi_start_h = round(bottom_rois[2] * spatial_scale_);
        // int roi_end_w = round(bottom_rois[3] * spatial_scale_);
        // int roi_end_h = round(bottom_rois[4] * spatial_scale_);

        int j = indices[r + 1];
        if (debug_info_)
          LOG(INFO) << "r: " << r << " j: " << j
                    << " scores: " << scores[j * num_class_ + c];

        Dtype w = std::min(rois[i * 5 + 3], rois[j * 5 + 3]) -
                  std::max(rois[i * 5 + 1], rois[j * 5 + 1]) + 1;
        Dtype h = std::min(rois[i * 5 + 4], rois[j * 5 + 4]) -
                  std::max(rois[i * 5 + 2], rois[j * 5 + 2]) + 1;

        if (w <= 0 || h <= 0) continue;
        Dtype s_j = (rois[j * 5 + 3] - rois[j * 5 + 1] + 1) *
                    (rois[j * 5 + 4] - rois[j * 5 + 2] + 1);
        Dtype ovlp = w * h / (s_i + s_j - w * h);

        if (debug_info_) LOG(INFO) << "ovlp: " << ovlp;
        if (ovlp < min_overlap_) continue;

        diff = diff_.mutable_cpu_data() + diff_.offset(n, c, r, 0);
        caffe_sub(feature_dim_, feature + j * feature_dim_,
                  feature + i * feature_dim_, diff);

        Dtype dot = caffe_cpu_dot(feature_dim_, diff, diff);
        Dtype prob;
        if (is_sigmoid_)
          prob = sigmoid(scores[j * num_class_ + c]);
        else
          prob = scores[j * num_class_ + c];

        loss[0] += dot * prob / 2;
        // loss[0] += dot * prob  / 2;
        if (debug_info_) LOG(INFO) << "dot: " << dot;

        bbox_idx[bbox_idx_.offset(n, c, r, 0)] = j;
        bbox_prob[bbox_prob_.offset(n, c, r, 0)] = prob;
        total_num_this_++;
        total_num_++;
      }
    }
  }

  total_loss_ += loss[0];
  total_im_ += num_im_;
  if (total_im_ % 1280 == 0) {
    LOG(INFO) << "total_num_: " << total_num_ << " total_loss_: " << total_loss_
              << " ave_loss: " << total_loss_ / total_num_;
    total_im_ = 0;
    total_num_ = 0;
    total_loss_ = 0;
  }

  if (debug_info_) LOG(INFO) << "total_num_this_: " << total_num_this_;

  if (total_num_this_ > 0) loss[0] /= total_num_this_;
}

template <typename Dtype>
void SmoothFeatureLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // feature		#roi	c	h	w
  // rois		#roi	5	1	1
  // score		#roi	#class	1	1
  // label		#im	#class

  if (debug_info_) LOG(INFO) << "SmoothFeatureLossLayer Backward_cpu";

  if (!propagate_down[0]) {
    return;
  }

  caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
  if (total_num_this_ <= 0) {
    return;
  }

  const Dtype alpha = top[0]->cpu_diff()[0] / total_num_this_;
  // const Dtype alpha = top[0]->cpu_diff()[0];

  const Dtype* scores = bottom[2]->cpu_data();
  const Dtype* label = bottom[3]->cpu_data();

  const int* max_bbox_idx = max_bbox_idx_.cpu_data();
  const int* bbox_idx = bbox_idx_.cpu_data();
  const Dtype* bbox_prob = bbox_prob_.cpu_data();
  const Dtype* diff;

  Dtype* feature_diff = bottom[0]->mutable_cpu_diff();

  for (int n = 0; n < num_im_; ++n) {
    CHECK_EQ(n, 0) << "current only support one image per minibatch";
    for (int c = 0; c < num_class_; ++c) {
      if (label[n * num_class_ + c] == 0) continue;
      int i = max_bbox_idx[max_bbox_idx_.offset(n, c, 0, 0)];
      if (debug_info_)
        LOG(INFO) << "c: " << c << " i: " << i
                  << " scores: " << scores[i * num_class_ + c];
      for (int r = 0; r < max_num_bbox_; ++r) {
        int j = bbox_idx[bbox_idx_.offset(n, c, r, 0)];
        if (j == -1) continue;
        Dtype prob = bbox_prob[bbox_prob_.offset(n, c, r, 0)];
        if (debug_info_)
          LOG(INFO) << "r: " << r << " j: " << j << " prob: " << prob;

        diff = diff_.cpu_data() + diff_.offset(n, c, r, 0);
        caffe_cpu_axpby(feature_dim_, +1 * alpha * prob, diff, Dtype(1),
                        feature_diff + j * feature_dim_);
        caffe_cpu_axpby(feature_dim_, -1 * alpha * prob, diff, Dtype(1),
                        feature_diff + i * feature_dim_);
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SmoothFeatureLossLayer);
#endif

INSTANTIATE_CLASS(SmoothFeatureLossLayer);
REGISTER_LAYER_CLASS(SmoothFeatureLoss);

}  // namespace caffe

#include <vector>
#include <cfloat>

#include "caffe/layers/weighted_center_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void WeightedCenterLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LOG_IF(INFO, debug_info_) << "    [Forward] ";
  const Dtype* label_data = bottom[1]->cpu_data();
  const Dtype* score_data = bottom[2]->cpu_data();
  const Dtype* feature_data = bottom[0]->gpu_data();
  const Dtype* center_data = this->blobs_[0]->gpu_data();

  Dtype* diff_data = diff_.mutable_gpu_data();
  Dtype* diff_diff = diff_.mutable_gpu_diff();

  Dtype dot = 0;
  for (int c = 0; c < num_class_; ++c) {
    center_selector_[c] = -1;
    loss_scale_[c] = 0;
    if (label_data[c] != 1) continue;
    LOG_IF(INFO, debug_info_) << "class: " << c;
    num_gt_class_++;

    Dtype c_dot = FLT_MAX;
    for (int m = 0; m < num_center_; ++m) {
      Dtype cm_dot = 0;
      Dtype scale = 0;
      for (int r = 0; r < num_roi_; ++r) {
        caffe_gpu_sub(dim_, feature_data + r * dim_,
                      center_data + (c * num_center_ + m) * dim_,
                      diff_diff + (c * num_roi_ + r) * dim_);
        Dtype cmr_dot;
        caffe_gpu_dot(dim_, diff_diff + (c * num_roi_ + r) * dim_,
                      diff_diff + (c * num_roi_ + r) * dim_, &cmr_dot);
        cm_dot += cmr_dot * score_data[r * num_class_ + c];
        scale += score_data[r * num_class_ + c];
      }
      if (cm_dot < c_dot) {
        caffe_copy(num_roi_ * dim_, diff_diff + (c * num_roi_ + 0) * dim_,
                   diff_data + (c * num_roi_ + 0) * dim_);
        c_dot = cm_dot;
        center_selector_[c] = m;
        loss_scale_[c] = scale;
      }
    }
    num_update_class_[c][center_selector_[c]]++;
    accum_update_class_[c][center_selector_[c]]++;
    dot += c_dot / loss_scale_[c];
  }

  Dtype loss = dot / num_gt_class_ / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;

  accum_loss_ += loss;
  total_iter_++;
  if (total_iter_ % display_ == 0) {
    LOG(INFO) << this->layer_param().name() << " #iter_: " << total_iter_
              << " #loss_: " << accum_loss_
              << " AVE loss: " << accum_loss_ / display_;
    accum_loss_ = 0;

    for (int c = 0; c < num_class_; ++c) {
      std::cout << "(";
      for (int m = 0; m < num_center_; ++m) {
        std::cout << accum_update_class_[c][m] << " ";
        accum_update_class_[c][m] = 0;
      }
      std::cout << "\b)";
    }
    std::cout << std::endl;
  }
  LOG_IF(INFO, debug_info_) << this->blobs_[0]->asum_data() << " "
                            << diff_.asum_data();
  LOG_IF(INFO, debug_info_) << this->blobs_[0]->asum_diff() << " "
                            << diff_.asum_diff();
}

template <typename Dtype>
void WeightedCenterLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  LOG_IF(INFO, debug_info_) << "    [Backward] ";
  if (!propagate_down[0]) return;

  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());

  const Dtype* label_data = bottom[1]->cpu_data();
  const Dtype* score_data = bottom[2]->cpu_data();
  const Dtype* diff_data = diff_.gpu_data();
  LOG_IF(INFO, debug_info_) << " weight: " << top[0]->cpu_diff()[0];

  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* center_diff = this->blobs_[0]->mutable_gpu_diff();

  for (int c = 0; c < num_class_; ++c) {
    if (label_data[c] != 1) continue;
    LOG_IF(INFO, debug_info_) << "class: " << c;

    const Dtype alpha = top[0]->cpu_diff()[0] / num_gt_class_ / loss_scale_[c];
    for (int r = 0; r < num_roi_; ++r) {
      // feature diff
      caffe_gpu_axpby(dim_, alpha, diff_data + (c * num_roi_ + r) * dim_,
                      Dtype(1), bottom_diff + r * dim_);

      // center diff
      caffe_gpu_axpby(
          dim_, Dtype(-1) * score_data[r * num_class_ + c] / loss_scale_[c],
          diff_data + (c * num_roi_ + r) * dim_, Dtype(1),
          center_diff + (c * num_center_ + center_selector_[c]) * dim_);
    }
  }

  // update center
  if (total_iter_ % update_ == 0) {
    for (int c = 0; c < num_class_; ++c) {
      for (int m = 0; m < num_center_; ++m) {
        caffe_gpu_axpy(
            dim_, lr_ * Dtype(-1) / (num_update_class_[c][m] + 1),
            this->blobs_[0]->gpu_diff() + (c * num_center_ + m) * dim_,
            this->blobs_[0]->mutable_gpu_data() + (c * num_center_ + m) * dim_);
        num_update_class_[c][m] = 0;
      }
    }
    caffe_gpu_set(this->blobs_[0]->count(), Dtype(0),
                  this->blobs_[0]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(WeightedCenterLossLayer);

}  // namespace caffe

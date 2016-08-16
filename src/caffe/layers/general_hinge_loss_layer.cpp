#include <algorithm>
#include <vector>

#include "caffe/layers/general_hinge_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GeneralHingeLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);

  display_ = this->layer_param_.general_hinge_loss_param().display();
  ignore_label_ = this->layer_param_.general_hinge_loss_param().ignore_label();
}

template <typename Dtype>
void GeneralHingeLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  // int dim = count / num;

  caffe_copy(count, bottom_data, bottom_diff);
  for (int i = 0; i < count; ++i) {
    if (i % num == ignore_label_) {
      bottom_diff[i] = 0;
      continue;
    }
    if (label[i] == 1)
      bottom_diff[i] = std::max(Dtype(0), 1 - bottom_diff[i]);
    else
      bottom_diff[i] = std::max(Dtype(0), 1 + bottom_diff[i]);
  }

  // for (int i = 0; i < num; ++i) {
  // for (int j = 0; j < dim; ++j) {
  // bottom_diff[i * dim + j] = std::max(
  // Dtype(0), 1 + bottom_diff[i * dim + j]);
  //}
  //}

  Dtype* loss = top[0]->mutable_cpu_data();
  switch (this->layer_param_.general_hinge_loss_param().norm()) {
    case GeneralHingeLossParameter_Norm_L1:
      loss[0] = caffe_cpu_asum(count, bottom_diff) / num;
      break;
    case GeneralHingeLossParameter_Norm_L2:
      loss[0] = caffe_cpu_dot(count, bottom_diff, bottom_diff) / num;
      break;
    default:
      LOG(FATAL) << "Unknown Norm";
  }

  total_loss_ += loss[0] / num;
  total_iter_++;
  if (total_iter_ % display_ == 0) {
    LOG(INFO) << this->layer_param().name() << " total_iter_: " << total_iter_
              << " total_loss_: " << total_loss_
              << " AVE loss: " << total_loss_ / total_iter_;
    total_loss_ = 0;
    total_iter_ = 0;
  }

  is_vis_ = false;
  // is_vis_=true;
  if (is_vis_) {
    LOG(INFO) << "###################forward bottom[0] "
                 "data################################";
    display_blob(bottom[0]);
    LOG(INFO) << "###################forward bottom[1] "
                 "data################################";
    display_blob(bottom[1]);
    LOG(INFO) << "###################forward top[0] "
                 "data###################################";
    display_blob(top[0]);
  }
}

template <typename Dtype>
void GeneralHingeLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* label = bottom[1]->cpu_data();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    // int dim = count / num;

    for (int i = 0; i < count; ++i) {
      if (i % num == ignore_label_) {
        bottom_diff[i] = 0;
        continue;
      }
      if (label[i] == 1) bottom_diff[i] *= -1;
    }

    // for (int i = 0; i < num; ++i) {
    // bottom_diff[i * dim + static_cast<int>(label[i])] *= -1;
    //}

    const Dtype loss_weight = top[0]->cpu_diff()[0];
    switch (this->layer_param_.general_hinge_loss_param().norm()) {
      case GeneralHingeLossParameter_Norm_L1:
        caffe_cpu_sign(count, bottom_diff, bottom_diff);
        caffe_scal(count, loss_weight / num, bottom_diff);
        break;
      case GeneralHingeLossParameter_Norm_L2:
        caffe_scal(count, loss_weight * 2 / num, bottom_diff);
        break;
      default:
        LOG(FATAL) << "Unknown Norm";
    }
  }
  if (is_vis_) {
    LOG(INFO) << "###################backward bottom[0] "
                 "diff#################################";
    display_blob(bottom[0], false);
  }
}

INSTANTIATE_CLASS(GeneralHingeLossLayer);
REGISTER_LAYER_CLASS(GeneralHingeLoss);

}  // namespace caffe

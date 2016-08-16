#include <vector>

#include "caffe/layers/cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  CrossEntropyLossParameter this_layer_param =
      this->layer_param_.cross_entropy_loss_param();
  display_ = this_layer_param.display();
  ignore_label_ = this_layer_param.ignore_label();

  total_loss_ = 0;
  total_iter_ = 0;
  total_ignore_num_ = 0;
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                           const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);

  count_ = bottom[0]->count();
  num_ = bottom[0]->num();
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
      << "CROSS_ENTROPY_LOSS layer inputs must have the same count.";
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();

  Dtype loss = 0;
  if (ignore_label_ >= 0) {
    for (int i = 0; i < count_; ++i) {
      if (i % num_ == ignore_label_) continue;
      Dtype prob = std::max(input_data[i], Dtype(kLOG_THRESHOLD));
      Dtype one_prob = std::max(1 - input_data[i], Dtype(kLOG_THRESHOLD));
      loss -= (target[i] * log(prob) + (1 - target[i]) * log(one_prob));
    }
  } else {
    for (int i = 0; i < count_; ++i) {
      if (target[i] == -1) {
        total_ignore_num_++;
        continue;
      }
      Dtype prob = std::max(input_data[i], Dtype(kLOG_THRESHOLD));
      Dtype one_prob = std::max(1 - input_data[i], Dtype(kLOG_THRESHOLD));
      loss -= (target[i] * log(prob) + (1 - target[i]) * log(one_prob));
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / num_;

  total_loss_ += loss / num_;
  total_iter_++;
  if (total_iter_ % display_ == 0) {
    LOG(INFO) << this->layer_param().name() << " #iter_: " << total_iter_
              << " #ignore: " << total_ignore_num_
              << " #loss_: " << total_loss_
              << " AVE loss: " << total_loss_ / total_iter_;
    total_loss_ = 0;
    total_iter_ = 0;
    total_ignore_num_ = 0;
  }
}

template <typename Dtype>
void CrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }

  if (!propagate_down[0]) {
    return;
  }
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  const Dtype scale = top[0]->cpu_diff()[0] / num_;
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  if (ignore_label_ >= 0) {
    for (int i = 0; i < count_; ++i) {
      if (i % num_ == ignore_label_) {
        bottom_diff[i] = 0;
        continue;
      }
      Dtype prob = std::max(input_data[i], Dtype(kLOG_THRESHOLD));
      Dtype one_prob = std::max(1 - input_data[i], Dtype(kLOG_THRESHOLD));
      bottom_diff[i] = std::min(
          scale * (-1 * target[i] / prob - (-1) * (1 - target[i]) / one_prob),
          Dtype(kDIFF_THRESHOLD));
    }
  } else {
    for (int i = 0; i < count_; ++i) {
      if (target[i] == -1) {
        bottom_diff[i] = 0;
        continue;
      }
      Dtype prob = std::max(input_data[i], Dtype(kLOG_THRESHOLD));
      Dtype one_prob = std::max(1 - input_data[i], Dtype(kLOG_THRESHOLD));
      bottom_diff[i] = std::min(
          scale * (-1 * target[i] / prob - (-1) * (1 - target[i]) / one_prob),
          Dtype(kDIFF_THRESHOLD));
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(CrossEntropyLossLayer, Backward);
#endif

INSTANTIATE_CLASS(CrossEntropyLossLayer);
REGISTER_LAYER_CLASS(CrossEntropyLoss);

}  // namespace caffe

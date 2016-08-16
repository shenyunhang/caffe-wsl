#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/log_sum_exp_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
Dtype max_element_(const Dtype *in, const int count) {
  Dtype max_value = -FLT_MAX;
  for (int i = 0; i < count; ++i) {
    if (max_value < (*in)) {
      max_value = (*in);
    }
    ++in;
  }
  return max_value;
}

template <typename Dtype>
void LogSumExpLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                        const vector<Blob<Dtype> *> &top) {
  // rx
  exp_.CopyFrom(*bottom[0], false, false);
  exp_.scale_data(r_);

  // rx^star
  Dtype max_value = max_element_(exp_.cpu_data(), count_);

  // rx-rx^star
  caffe_gpu_add_scalar(count_, Dtype(-1.0) * (max_value),
                       exp_.mutable_gpu_data());

  // exp(rx-rx^star)
  caffe_gpu_exp(count_, exp_.gpu_data(), exp_.mutable_gpu_data());

  const Dtype *exp_cpu = exp_.cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  Dtype *sum_exp_mcpu = sum_exp_.mutable_cpu_data();
  for (int c = 0; c < channels_; ++c) {
    Dtype sum_c = 0;
    for (int n = 0; n < num_; ++n) {
      sum_c += caffe_cpu_asum(inner_count_, exp_cpu + exp_.offset(n, c, 0, 0));
    }
    sum_c = std::max(sum_c, Dtype(kLOG_THRESHOLD));
    sum_exp_mcpu[c] = sum_c;
    top_data[c] = ((max_value) + log(sum_c) - log(num_ * inner_count_)) / r_;
    /*top_data[c] = (rx_star_cpu[c] + log(sum) - log(0)) / r_;*/
    /*top_data[c] = (rx_star_cpu[c] + log(sum) - log(num_roi)) / r_ * num_roi;*/
  }

  if (debug_info_) {
    LOG(INFO) << "exp_: " << exp_.asum_data() / exp_.count();
    LOG(INFO) << "sum_exp_: " << sum_exp_.asum_data() / sum_exp_.count();
    LOG(INFO) << "max_value: " << (max_value);
  }
}

template <typename Dtype>
void LogSumExpLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
                                         const vector<bool> &propagate_down,
                                         const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[0] == false) {
    return;
  }

  caffe_copy(count_, exp_.gpu_data(), bottom[0]->mutable_gpu_diff());

  const Dtype *top_diff = top[0]->cpu_diff();
  const Dtype *sum_exp = sum_exp_.cpu_data();
  Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();

  for (int c = 0; c < channels_; ++c) {
    for (int n = 0; n < num_; ++n) {
      caffe_scal(inner_count_, top_diff[c] / sum_exp[c],
                 bottom_diff + bottom[0]->offset(n, c, 0, 0));
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(LogSumExpLayer);

}  // namespace caffe

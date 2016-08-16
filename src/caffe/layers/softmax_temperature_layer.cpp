#include <vector>

#include "caffe/layers/softmax_temperature_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxTemperatureLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  temperature_ = this->layer_param_.softmax_temperature_param().temperature();
  is_append_ = this->layer_param_.softmax_temperature_param().is_append();
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());

  LOG(INFO) << "----------------------------------------------";
  LOG(INFO) << "temperature_: " << temperature_;
  LOG(INFO) << "softmax_axis_: " << softmax_axis_;
  LOG(INFO) << "is_append_: " << is_append_;
  LOG(INFO) << "----------------------------------------------";

  softmax_input_->ReshapeLike(*bottom[0]);
  softmax_output_->ReshapeLike(*bottom[0]);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(softmax_input_.get());
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(softmax_output_.get());

  // we need call setup as net.cpp do to each layer.
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
}

template <typename Dtype>
void SoftmaxTemperatureLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);

  vector<int> softmax_shape;
  softmax_shape.push_back(bottom[0]->shape(0));
  softmax_shape.push_back(bottom[0]->shape(1));
  if (is_append_) softmax_shape[softmax_axis_]++;
  softmax_input_->Reshape(softmax_shape);
  softmax_output_->Reshape(softmax_shape);

  // It seems Forward function will call Reshape function
  // softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
}

template <typename Dtype>
void SoftmaxTemperatureLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (is_append_)
    Copy_blob(bottom[0], softmax_input_.get(), false);
  else
    softmax_input_->CopyFrom(*bottom[0], false, false);
  caffe_scal(softmax_input_->count(), temperature_,
             softmax_input_->mutable_cpu_data());
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  if (is_append_)
    Copy_blob(softmax_output_.get(), top[0], false);
  else
    top[0]->CopyFrom(*softmax_output_, false, false);
}

template <typename Dtype>
void SoftmaxTemperatureLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (is_append_)
    Copy_blob(top[0], softmax_output_.get(), true);
  else
    softmax_output_->CopyFrom(*top[0], true, false);
  softmax_layer_->Backward(softmax_top_vec_, propagate_down,
                           softmax_bottom_vec_);
  caffe_scal(softmax_input_->count(), temperature_,
             softmax_input_->mutable_cpu_diff());
  if (is_append_)
    Copy_blob(softmax_input_.get(), bottom[0], true);
  else
    bottom[0]->CopyFrom(*softmax_input_, true, false);
}

template <typename Dtype>
void SoftmaxTemperatureLayer<Dtype>::Copy_blob(const Blob<Dtype>* input_blob,
                                               Blob<Dtype>* output_blob,
                                               const bool diff) {
  const int i_h = input_blob->shape(0);
  const int i_w = input_blob->shape(1);
  const int o_h = output_blob->shape(0);
  const int o_w = output_blob->shape(1);

  const Dtype* in_data;
  Dtype* out_data;
  if (diff) {
    in_data = input_blob->cpu_diff();
    out_data = output_blob->mutable_cpu_diff();
  } else {
    in_data = input_blob->cpu_data();
    out_data = output_blob->mutable_cpu_data();
  }

  for (int h = 0; h < o_h; ++h) {
    for (int w = 0; w < o_w; ++w) {
      if (h >= i_h || w >= i_w) {
        out_data[h * o_w + w] = 0;
      } else {
        out_data[h * o_w + w] = in_data[h * i_w + w];
      }
    }
  }

}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(SoftmaxTemperatureLayer, Backward);
#endif

INSTANTIATE_CLASS(SoftmaxTemperatureLayer);
REGISTER_LAYER_CLASS(SoftmaxTemperature);

}  // namespace caffe

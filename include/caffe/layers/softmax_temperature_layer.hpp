#ifndef CAFFE_SOFTMAX_TEMPERATURE_LAYER_HPP_
#define CAFFE_SOFTMAX_TEMPERATURE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/softmax_layer.hpp"

namespace caffe {

template <typename Dtype>
class SoftmaxTemperatureLayer : public Layer<Dtype> {
 public:
  explicit SoftmaxTemperatureLayer(const LayerParameter& param)
      : Layer<Dtype>(param),
        softmax_layer_(new SoftmaxLayer<Dtype>(param)),
        softmax_input_(new Blob<Dtype>()),
        softmax_output_(new Blob<Dtype>()) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SoftmaxTemperature"; }

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

  void Copy_blob(const Blob<Dtype>* input_blob, Blob<Dtype>* output_blob,
                 bool diff);

  /// The internal SoftmaxLayer
  shared_ptr<SoftmaxLayer<Dtype> > softmax_layer_;
  /// softmax_input stores the input of the SoftmaxLayer.
  shared_ptr<Blob<Dtype> > softmax_input_;
  /// softmax_output stores the output of the SoftmaxLayer.
  shared_ptr<Blob<Dtype> > softmax_output_;
  /// bottom vector holder to call the underlying SoftmaxLayer::Forward
  vector<Blob<Dtype>*> softmax_bottom_vec_;
  /// top vector holder to call the underlying SoftmaxLayer::Forward
  vector<Blob<Dtype>*> softmax_top_vec_;

  Dtype temperature_;
  int softmax_axis_;
  bool is_append_;
};

}  // namespace caffe

#endif  // CAFFE_SOFTMAX_TEMPERATURE_LAYER_HPP_

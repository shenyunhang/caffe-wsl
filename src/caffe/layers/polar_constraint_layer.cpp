#include <vector>

#include "caffe/layers/polar_constraint_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PolarConstraintLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  // data 	#roi 	#class 	1 	1
  // fliter	#roi 	#class 	1 	1
  // [0,1) denote the score
  // 1 denote in postive bag

  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))
      << "POLAR_CONSTRAINT layer inputs must have the same shape.";
  CHECK_EQ(bottom[0]->shape(1), bottom[1]->shape(1))
      << "POLAR_CONSTRAINT layer inputs must have the same shape.";
  CHECK_EQ(bottom[0]->count(), bottom[1]->count())
      << "POLAR_CONSTRAINT layer inputs must have the same count.";

  polar_ = this->layer_param_.polar_constraint_param().polar();
}

template <typename Dtype>
void PolarConstraintLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  count_ = bottom[0]->count();
  channels_ = bottom[0]->channels();

  fliter.ReshapeLike(*bottom[0]);
  top[0]->ReshapeLike(*bottom[0]);

  if (polar_) {
  } else {
     //vector<int> top1_shape;
     //top1_shape.push_back(1);
     //top1_shape.push_back(channels_);
     //top[1]->Reshape(top1_shape);

    //top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void PolarConstraintLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void PolarConstraintLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(PolarConstraintLayer, Backward);
#endif

INSTANTIATE_CLASS(PolarConstraintLayer);
REGISTER_LAYER_CLASS(PolarConstraint);

}  // namespace caffe

#include <vector>

#include "caffe/layers/pooling_sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
		void PoolingSigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
				const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
				const vector<Blob<Dtype>*>& bottom) {
			if (propagate_down[1]) {
				LOG(FATAL) << this->type()
					<< " Layer cannot backpropagate to label inputs.";
			}
			if (propagate_down[0]) {
				// First, compute the diff
				const int count = bottom[0]->count();
				const int num = bottom[1]->num();
				const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
				/*const Dtype* target = bottom[1]->gpu_data();*/
				const Dtype* pseudo_target = pseudo_target_.gpu_data();
				Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
				caffe_copy(count, sigmoid_output_data, bottom_diff);
				/*caffe_gpu_axpy(count, Dtype(-1), target, bottom_diff);*/
				caffe_gpu_axpy(count, Dtype(-1), pseudo_target, bottom_diff);
				// Scale down gradient
				const Dtype loss_weight = top[0]->cpu_diff()[0];
				caffe_gpu_scal(count, loss_weight / num, bottom_diff);

			}

			if(is_vis_){
				std::cout<<"###################backward bottom[0] diff################################"<<std::endl;
				display_blob(bottom[0],false);
				std::cout<<"###################backward sigmoid out0] data############################"<<std::endl;
				display_blob(&*sigmoid_output_);
				std::cout<<"###################backward pseudo_target_ data###########################"<<std::endl;
				display_blob(&pseudo_target_);
			}
		}


	INSTANTIATE_LAYER_GPU_BACKWARD(PoolingSigmoidCrossEntropyLossLayer);

}  // namespace caffe

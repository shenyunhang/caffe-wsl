#include <vector>
#include <cfloat>
#include <stdio.h>

#include "caffe/layers/pooling_sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
		void PoolingSigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
				const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
			LossLayer<Dtype>::LayerSetUp(bottom, top);
			sigmoid_bottom_vec_.clear();
			sigmoid_bottom_vec_.push_back(bottom[0]);
			sigmoid_top_vec_.clear();
			sigmoid_top_vec_.push_back(sigmoid_output_.get());
			sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
		}

	template <typename Dtype>
		void PoolingSigmoidCrossEntropyLossLayer<Dtype>::Reshape(
				const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
			//LossLayer<Dtype>::Reshape(bottom, top);
			// 用下面两替代LossLayer<Dtype>::Reshape(bottom, top) 以此跳过里面的CHECK
			vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
			top[0]->Reshape(loss_shape);

			// score	#roi	#class	1	1
			// label	#im	#class 	1	1
			// rois		#roi	5	1	1
			// fliter	#roi	#class	1	1

			CHECK_EQ(bottom[0]->num(), bottom[2]->num()) <<
				"Pooling SIGMOID_CROSS_ENTROPY_LOSS layer: first and third blob must have the same num.";
			CHECK_EQ(bottom[0]->channels(), bottom[1]->channels()) <<
				"Pooling SIGMOID_CROSS_ENTROPY_LOSS layer: first and second blob must have the same channels.";
			CHECK_EQ(bottom[0]->num(), bottom[3]->num()) <<
				"Pooling SIGMOID_CROSS_ENTROPY_LOSS layer: first and forth blob must have the same num.";
			CHECK_EQ(bottom[0]->channels(), bottom[3]->channels()) <<
				"Pooling SIGMOID_CROSS_ENTROPY_LOSS layer: first and forth blob must have the same channels.";

			CHECK_EQ(bottom[0]->count(), bottom[0]->num()*bottom[0]->channels()) <<
				"Pooling SIGMOID_CROSS_ENTROPY_LOSS layer: first blob height and width must 1.";
			CHECK_EQ(bottom[1]->count(), bottom[1]->num()*bottom[1]->channels()) <<
				"Pooling SIGMOID_CROSS_ENTROPY_LOSS layer: second blob height and width must 1.";
			CHECK_EQ(bottom[2]->count(), bottom[2]->num()*bottom[2]->channels()) <<
				"Pooling SIGMOID_CROSS_ENTROPY_LOSS layer: third blob height and width must 1.";
			CHECK_EQ(bottom[3]->count(), bottom[3]->num()*bottom[3]->channels()) <<
				"Pooling SIGMOID_CROSS_ENTROPY_LOSS layer: forth blob height and width must 1.";


			sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);

			max_idx_.Reshape(bottom[1]->num(),bottom[1]->channels(),1,1);
			max_idx_all_.Reshape(bottom[1]->num(),bottom[1]->channels(),1,1);
			pseudo_target_.ReshapeLike(*bottom[0]);

		}

	template <typename Dtype>
		void PoolingSigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
				const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
			// The forward pass computes the sigmoid outputs.
			sigmoid_bottom_vec_[0] = bottom[0];
			sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
			// Compute the loss (negative log likelihood)

			//const int count = bottom[1]->count();
			const int num_roi = bottom[0]->num();
			const int num_im = bottom[1]->num();
			const int num_class = bottom[1]->channels();

			// Stable version of loss computation from input data
			const Dtype* input_data = bottom[0]->cpu_data();
			const Dtype* target = bottom[1]->cpu_data();

			//const Dtype* region_number = bottom[2]->cpu_data();
			const Dtype* bottom_rois = bottom[2]->cpu_data();

			const Dtype* fliter = bottom[3]->cpu_data();

			int* argmax_data=max_idx_.mutable_cpu_data();
			int* argmax_data_all=max_idx_all_.mutable_cpu_data();
			Dtype* pseudo_target=pseudo_target_.mutable_cpu_data();
			const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
			caffe_copy(bottom[0]->count(),sigmoid_output_data,pseudo_target);

			Dtype loss = 0;
			int region_num_used=0;

			for (int ph = 0; ph < num_im ; ++ph) {
				int hstart=region_num_used;
				int hend=hstart;
				for(;hend<num_roi;++hend){
					const Dtype* rois=bottom_rois+bottom[2]->offset(hend);
					int im_index=rois[0];
					if(im_index!=ph){
						break;
					}
				}
				CHECK_GT(hend,hstart)<<"hend should great than hstart";

				for(int pw=0; pw<num_class; ++pw){
					int wstart=pw;
					int wend=wstart+1;

					const int pool_index=ph*num_class+pw;

					Dtype in_max=-FLT_MAX;
					Dtype in_max_all=-FLT_MAX;
					Dtype pos_sum=0;
					int pos_num=0;
					for(int h=hstart;h<hend;++h){
						for(int w=wstart;w<wend;++w){
							const int index= h*num_class+w;

							// find the max value in spite with mask
							if(input_data[index]>in_max_all){
								in_max_all=input_data[index];
								argmax_data_all[pool_index]=index;
							}

							if(fliter[index]==Dtype(0)) continue;

							// find the max value
							if(input_data[index]>in_max){
								in_max=input_data[index];
								argmax_data[pool_index]=index;
							}

							// sum all postive value
							if(input_data[index]>0){
								pos_sum+=input_data[index];
								pos_num++;
								pseudo_target[index]=target[pool_index];
							}
						}
					}

					if(pos_num>0){
						// ave pooling
						CHECK_NE(pos_sum,0)<<"pos_sum should large than zero";
						CHECK_NE(pos_num,0)<<"pos_num should large than zero";

						Dtype pos_ave=1.0 * pos_sum / pos_num;
						loss -= pos_ave * (target[pool_index] - (pos_ave >= 0)) -
							log(1 + exp(pos_ave - 2 * pos_ave * (pos_ave >= 0)));

					}else{
						// max pooling
						CHECK_EQ(pos_sum,0)<<"pos_sum should be zero";
						CHECK_EQ(pos_num,0)<<"pos_num should be zero";
						if(in_max==-FLT_MAX){
							//CHECK_NE(in_max_all,-FLT_MAX)<<"in_max_all should not be -FLT_MAX";

							pseudo_target[argmax_data_all[pool_index]]=target[pool_index];
							loss -= in_max_all * (target[pool_index] - (in_max_all >= 0)) -
								log(1 + exp(in_max_all - 2 * in_max_all * (in_max_all >= 0)));
						}else{
							pseudo_target[argmax_data[pool_index]]=target[pool_index];
							loss -= in_max * (target[pool_index] - (in_max >= 0)) -
								log(1 + exp(in_max - 2 * in_max * (in_max >= 0)));

						}
					}

				}
				region_num_used=hend;

				//loss -= input_data[i] * (target[i] - (input_data[i] >= 0)) -
				//log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
			}
			CHECK_EQ(region_num_used,num_roi)<<"region_num_used should equal num_roi";

			top[0]->mutable_cpu_data()[0] = loss / num_im;

			is_vis_=false;
			//is_vis_=true;
			if(is_vis_){
				std::cout<<"###################forward bottom[0] data###############################"<<std::endl;
				display_blob(bottom[0]);
				std::cout<<"###################forward sigmoid out0] data###########################"<<std::endl;
				display_blob(&*sigmoid_output_);
				std::cout<<"###################forward bottom[1] data###############################"<<std::endl;
				display_blob(bottom[1]);
				std::cout<<"###################forward top[0] data##################################"<<std::endl;
				display_blob(top[0]);
				std::cout<<"###################forward pseudo_target_ data##########################"<<std::endl;
				display_blob(&pseudo_target_);
			}
		}

	template <typename Dtype>
		void PoolingSigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
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
				const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
				//const Dtype* target = bottom[1]->cpu_data();
				const Dtype* pseudo_target = pseudo_target_.cpu_data();
				Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
				caffe_sub(count, sigmoid_output_data, pseudo_target, bottom_diff);
				// Scale down gradient
				const Dtype loss_weight = top[0]->cpu_diff()[0];
				caffe_scal(count, loss_weight / num, bottom_diff);
			}
		}


#ifdef CPU_ONLY
	STUB_GPU_BACKWARD(PoolingSigmoidCrossEntropyLossLayer, Backward);
#endif

	INSTANTIATE_CLASS(PoolingSigmoidCrossEntropyLossLayer);
	REGISTER_LAYER_CLASS(PoolingSigmoidCrossEntropyLoss);

}  // namespace caffe

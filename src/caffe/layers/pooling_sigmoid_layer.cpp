#include <cmath>
#include <vector>

#include "caffe/layers/pooling_sigmoid_layer.hpp"

namespace caffe {

	template <typename Dtype>
		inline Dtype poolingsigmoid(Dtype x, Dtype fliter) {
			return 1. / (1. + exp(-x)) * fliter;
		}

	template <typename Dtype>
		void PoolingSigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
				const vector<Blob<Dtype>*>& top) {
			const Dtype* bottom_data = bottom[0]->cpu_data();
			Dtype* top_data = top[0]->mutable_cpu_data();
			const int count = bottom[0]->count();
			const Dtype* fliter_data=bottom[1]->gpu_data();
			for (int i = 0; i < count; ++i) {
				top_data[i] = poolingsigmoid(bottom_data[i], fliter_data[i]);
			}
		}

	template <typename Dtype>
		void PoolingSigmoidLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
				const vector<bool>& propagate_down,
				const vector<Blob<Dtype>*>& bottom) {
			if (propagate_down[0]) {
				const Dtype* top_data = top[0]->cpu_data();
				const Dtype* top_diff = top[0]->cpu_diff();
				Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
				const int count = bottom[0]->count();
				const Dtype* fliter_data=bottom[1]->gpu_data();
				for (int i = 0; i < count; ++i) {
					const Dtype sigmoid_x = top_data[i];
					bottom_diff[i] = top_diff[i] * sigmoid_x * (1. - sigmoid_x) * fliter_data[i];
				}
			}
		}

#ifdef CPU_ONLY
	STUB_GPU(PoolingSigmoidLayer);
#endif

	INSTANTIATE_CLASS(PoolingSigmoidLayer);
	REGISTER_LAYER_CLASS(PoolingSigmoid);


}  // namespace caffe

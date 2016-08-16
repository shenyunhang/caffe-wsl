#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/global_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void GlobalPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    threshold_ = this->layer_param_.global_pooling_param().threshold();
    LOG(INFO) << "----------------------------------------------";
    switch (this->layer_param_.global_pooling_param().pool()) {
    case GlobalPoolingParameter_PoolMethod_TSUM:
	LOG(INFO) << "Using threshold SUM pooling method.";
	LOG(INFO) << "threshold_: " << threshold_;
	break;
    case GlobalPoolingParameter_PoolMethod_TAVEMAX:
	LOG(INFO) << "Using threshold AVE and MAX pooling method.";
	LOG(INFO) << "threshold_: " << threshold_;
	break;
    default:
	LOG(FATAL) << "Unknown pooling method.";
    }
    LOG(INFO) << "INT_MAX: " << INT_MAX;
    LOG(INFO) << "INT_MIN: " << INT_MIN;
    LOG(INFO) << "----------------------------------------------";
}

template <typename Dtype>
void GlobalPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
				       << "corresponding to (num, channels, height, width)";
    top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
    mask_idx_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
}

template <typename Dtype>
void GlobalPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top)
{
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    const int top_count = top[0]->count();
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int height = bottom[0]->height();
    const int width = bottom[0]->width();

    int* mask = mask_idx_.mutable_cpu_data();
    caffe_set(top_count, -1, mask);
    // Different pooling methods. We explicitly do the switch outside the for
    // loop to save time, although this results in more code.
    switch (this->layer_param_.global_pooling_param().pool()) {
    case GlobalPoolingParameter_PoolMethod_TSUM:
	// Initialize
	caffe_set(top_count, Dtype(0), top_data);
	// The main loop
	for (int n = 0; n < num; ++n) {
	    for (int c = 0; c < channels; ++c) {
		const int pool_index = top[0]->offset(n, c);
		int pool_size = 0;
		Dtype all_sum = 0;
		for (int h = 0; h < height; ++h) {
		    for (int w = 0; w < width; ++w) {
			const int index = bottom[0]->offset(n, c, h, w);
			const Dtype in = bottom_data[index];
			all_sum += in;
			if (in < threshold_)
			    continue;
			top_data[pool_index] += in;
			pool_size++;
		    }
		}
		if (pool_size == 0) {
		    top_data[pool_index] = all_sum;
		}
		mask[pool_index] = pool_size;
	    }
	}
	break;
    case GlobalPoolingParameter_PoolMethod_TAVEMAX:
	// Initialize
	caffe_set(top_count, Dtype(0), top_data);
	// The main loop
	for (int n = 0; n < num; ++n) {
	    for (int c = 0; c < channels; ++c) {
		const int pool_index = top[0]->offset(n, c);
		int pool_size = 0;
		Dtype max_value = -FLT_MAX;
		int max_value_index = -1;
		for (int h = 0; h < height; ++h) {
		    for (int w = 0; w < width; ++w) {
			const int index = bottom[0]->offset(n, c, h, w);
			const Dtype in = bottom_data[index];
			if (in > max_value) {
			    max_value = in;
			    max_value_index = index;
			}
			if (in < threshold_)
			    continue;
			top_data[pool_index] += in;
			pool_size++;
		    }
		}
		if (pool_size == 0) {
		    top_data[pool_index] = max_value;
		    mask[pool_index] = max_value_index;

		    mask[pool_index] *= -1;
		} else {
		    top_data[pool_index] /= pool_size;
		    mask[pool_index] = pool_size;
		}
	    }
	}
	break;
    default:
	LOG(FATAL) << "Unknown pooling method.";
    }
}

template <typename Dtype>
void GlobalPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    if (!propagate_down[0]) {
	return;
    }
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();

    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int height = bottom[0]->height();
    const int width = bottom[0]->width();

    const int* mask = mask_idx_.cpu_data();
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

    // Different pooling methods. We explicitly do the switch outside the for
    // loop to save time, although this results in more codes.
    switch (this->layer_param_.global_pooling_param().pool()) {
    case GlobalPoolingParameter_PoolMethod_TSUM:
	// The main loop
	for (int n = 0; n < num; ++n) {
	    for (int c = 0; c < channels; ++c) {
		const int pool_index = top[0]->offset(n, c);
		const int pool_size = mask[pool_index];
		if (pool_size == 0) {
		    for (int h = 0; h < height; ++h) {
			for (int w = 0; w < width; ++w) {
			    const int index = bottom[0]->offset(n, c, h, w);
			    bottom_diff[index] = top_diff[pool_index];
			}
		    }
		} else {
		    for (int h = 0; h < height; ++h) {
			for (int w = 0; w < width; ++w) {
			    const int index = bottom[0]->offset(n, c, h, w);
			    if (bottom_data[index] < threshold_)
				continue;
			    bottom_diff[index] = top_diff[pool_index];
			}
		    }
		}
	    }
	}
	break;
    case GlobalPoolingParameter_PoolMethod_TAVEMAX:
	// The main loop
	for (int n = 0; n < num; ++n) {
	    for (int c = 0; c < channels; ++c) {
		const int pool_index = top[0]->offset(n, c);
		const int pool_size = mask[pool_index];
		if (pool_size > 0) {
		    for (int h = 0; h < height; ++h) {
			for (int w = 0; w < width; ++w) {
			    const int index = bottom[0]->offset(n, c, h, w);
			    if (bottom_data[index] < threshold_)
				continue;
			    bottom_diff[index] = top_diff[pool_index] / pool_size;
			}
		    }
		} else {
		    const int bottom_index = -1 * pool_size;
		    bottom_diff[bottom_index] = top_diff[pool_index];
		}
	    }
	}
	break;
    default:
	LOG(FATAL) << "Unknown pooling method.";
    }
}

#ifdef CPU_ONLY
STUB_GPU(GlobalPoolingLayer);
#endif

INSTANTIATE_CLASS(GlobalPoolingLayer);
REGISTER_LAYER_CLASS(GlobalPooling);

} // namespace caffe

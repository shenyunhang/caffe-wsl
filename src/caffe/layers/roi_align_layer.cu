#include <cfloat>

#include "caffe/layers/roi_align_layer.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__global__ void ROIAlignForward2(const int nthreads, const Dtype* bottom_data,
                                 const Dtype spatial_scale, const int channels,
                                 const int height, const int width,
                                 const int pooled_height,
                                 const int pooled_width,
                                 const Dtype* bottom_rois, Dtype* top_data,
                                 Dtype* argmax_data_w, Dtype* argmax_data_h) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    int roi_start_w = round(bottom_rois[1]);
    int roi_start_h = round(bottom_rois[2]);
    int roi_end_w = round(bottom_rois[3]);
    int roi_end_h = round(bottom_rois[4]);

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    Dtype bin_size_h =
        static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w =
        static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<Dtype>(ph) * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<Dtype>(pw) * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1) * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1) * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0),
                 static_cast<int>(static_cast<Dtype>(height) / spatial_scale));
    hend = min(max(hend + roi_start_h, 0),
               static_cast<int>(static_cast<Dtype>(height) / spatial_scale));
    wstart = min(max(wstart + roi_start_w, 0),
                 static_cast<int>(static_cast<Dtype>(width) / spatial_scale));
    wend = min(max(wend + roi_start_w, 0),
               static_cast<int>(static_cast<Dtype>(width) / spatial_scale));
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx_w = -1;
    int maxidx_h = -1;
    bottom_data += (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        // select four regular locations for bilinear interpolation
        Dtype sh = (static_cast<Dtype>(h) + 0.5) * spatial_scale - 0.5;
        Dtype sw = (static_cast<Dtype>(w) + 0.5) * spatial_scale - 0.5;
        int x_left = floor(sw);
        int x_right = ceil(sw);
        int y_top = floor(sh);
        int y_bottom = ceil(sh);

        int top_left_index = y_top * width + x_left;
        int top_right_index = y_top * width + x_right;
        int bottom_left_index = y_bottom * width + x_left;
        int bottom_right_index = y_bottom * width + x_right;

        bool is_top_left_in = x_left >= 0 && x_left <= width - 1 &&
                              y_top >= 0 && y_top <= height - 1;
        bool is_top_right_in = x_right >= 0 && x_right <= width - 1 &&
                               y_top >= 0 && y_top <= height - 1;
        bool is_bottom_left_in = x_left >= 0 && x_left <= width - 1 &&
                                 y_bottom >= 0 && y_bottom <= height - 1;
        bool is_bottom_right_in = x_right >= 0 && x_right <= width - 1 &&
                                  y_bottom >= 0 && y_bottom <= height - 1;

        Dtype val = 0;
        if (is_top_left_in)
          val += (1 - (sw - x_left)) * (1 - (sh - y_top)) *
                 bottom_data[top_left_index];
        if (is_top_right_in)
          val += (1 - (x_right - sw)) * (1 - (sh - y_top)) *
                 bottom_data[top_right_index];
        if (is_bottom_left_in)
          val += (1 - (sw - x_left)) * (1 - (y_bottom - sh)) *
                 bottom_data[bottom_left_index];
        if (is_bottom_right_in)
          val += (1 - (x_right - sw)) * (1 - (y_bottom - sh)) *
                 bottom_data[bottom_right_index];

        if (val > maxval) {
          maxval = val;
          maxidx_w = w;
          maxidx_h = h;
        }
      }
    }
    top_data[index] = maxval;
    argmax_data_w[index] = maxidx_w;
    argmax_data_h[index] = maxidx_h;
  }
}

template <typename Dtype>
__global__ void ROIAlignForward(const int nthreads, const Dtype* bottom_data,
                                const Dtype spatial_scale, const int channels,
                                const int height, const int width,
                                const int pooled_height, const int pooled_width,
                                const Dtype* bottom_rois, Dtype* top_data,
                                Dtype* argmax_data_w, Dtype* argmax_data_h) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    bottom_rois += n * 5;
    int roi_batch_ind = bottom_rois[0];
    Dtype roi_start_w = (bottom_rois[1] + 0.5) * spatial_scale - 0.5;
    Dtype roi_start_h = (bottom_rois[2] + 0.5) * spatial_scale - 0.5;
    Dtype roi_end_w = (bottom_rois[3] + 0.5) * spatial_scale - 0.5;
    Dtype roi_end_h = (bottom_rois[4] + 0.5) * spatial_scale - 0.5;

    // Force malformed ROIs to be 1x1
    Dtype roi_width = roi_end_w - roi_start_w;
    Dtype roi_height = roi_end_h - roi_start_h;

    Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

    Dtype hstart = static_cast<Dtype>(ph) * bin_size_h;
    Dtype wstart = static_cast<Dtype>(pw) * bin_size_w;
    Dtype hend = static_cast<Dtype>(ph + 1) * bin_size_h;
    Dtype wend = static_cast<Dtype>(pw + 1) * bin_size_w;

    // Add roi offsets and clip to input boundaries
    hstart = hstart + roi_start_h;
    hend = hend + roi_start_h;
    wstart = wstart + roi_start_w;
    wend = wend + roi_start_w;
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    // Define an empty pooling region to be zero
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    Dtype maxidx_w = -1;
    Dtype maxidx_h = -1;
    bottom_data += (roi_batch_ind * channels + c) * height * width;
    //for (Dtype h = hstart; h < hend; h += 0.5) {
      //for (Dtype w = wstart; w < wend; w += 0.5) {
    for (Dtype h = hstart+bin_size_h/4; h < hend; h += bin_size_h/2) {
      for (Dtype w = wstart+bin_size_w/4; w < wend; w += bin_size_w/2) {
        // select four regular locations for bilinear interpolation
        int x_left = floor(w);
        int x_right = ceil(w);
        int y_top = floor(h);
        int y_bottom = ceil(h);

        int top_left_index = y_top * width + x_left;
        int top_right_index = y_top * width + x_right;
        int bottom_left_index = y_bottom * width + x_left;
        int bottom_right_index = y_bottom * width + x_right;

        bool is_top_left_in = x_left >= 0 && x_left <= width - 1 &&
                              y_top >= 0 && y_top <= height - 1;
        bool is_top_right_in = x_right >= 0 && x_right <= width - 1 &&
                               y_top >= 0 && y_top <= height - 1;
        bool is_bottom_left_in = x_left >= 0 && x_left <= width - 1 &&
                                 y_bottom >= 0 && y_bottom <= height - 1;
        bool is_bottom_right_in = x_right >= 0 && x_right <= width - 1 &&
                                  y_bottom >= 0 && y_bottom <= height - 1;

        Dtype val = 0;
        if (is_top_left_in)
          val += (1 - (w - x_left)) * (1 - (h - y_top)) *
                 bottom_data[top_left_index];
        if (is_top_right_in)
          val += (1 - (x_right - w)) * (1 - (h - y_top)) *
                 bottom_data[top_right_index];
        if (is_bottom_left_in)
          val += (1 - (w - x_left)) * (1 - (y_bottom - h)) *
                 bottom_data[bottom_left_index];
        if (is_bottom_right_in)
          val += (1 - (x_right - w)) * (1 - (y_bottom - h)) *
                 bottom_data[bottom_right_index];
        if (val > maxval) {
          maxval = val;
          maxidx_w = w;
          maxidx_h = h;
        }
      }
    }
    top_data[index] = maxval;
    argmax_data_w[index] = maxidx_w;
    argmax_data_h[index] = maxidx_h;
  }
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* argmax_data_w = max_idx_w_.mutable_gpu_data();
  Dtype* argmax_data_h = max_idx_h_.mutable_gpu_data();
  int count = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ROIAlignForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
      (count, bottom_data, spatial_scale_, channels_, height_, width_,
       pooled_height_, pooled_width_, bottom_rois, top_data, argmax_data_w,
       argmax_data_h);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void ROIAlignBackward2(const int nthreads, const Dtype* top_diff,
                                  const Dtype* argmax_data_w,
                                  const Dtype* argmax_data_h,
                                  const int num_rois, const Dtype spatial_scale,
                                  const int channels, const int height,
                                  const int width, const int pooled_height,
                                  const int pooled_width, Dtype* bottom_diff,
                                  const Dtype* bottom_rois) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }

      int roi_start_w = round(bottom_rois[1]);
      int roi_start_h = round(bottom_rois[2]);
      int roi_end_w = round(bottom_rois[3]);
      int roi_end_h = round(bottom_rois[4]);

      // Skip if ROI doesn't include (h, w)
      const bool in_roi =
          (w >= floor((roi_start_w + 0.5) * spatial_scale - 0.5) &&
           w <= ceil((roi_start_h + 0.5) * spatial_scale - 0.5) &&
           h >= floor((roi_end_w + 0.5) * spatial_scale - 0.5) &&
           h <= ceil((roi_end_h + 0.5) * spatial_scale - 0.5));
      if (!in_roi) {
        continue;
      }

      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const Dtype* offset_top_diff = top_diff + offset;
      const Dtype* offset_argmax_data_w = argmax_data_w + offset;
      const Dtype* offset_argmax_data_h = argmax_data_h + offset;

      // Compute feasible set of pooled units that could have pooled
      // this bottom unit

      // Force malformed ROIs to be 1x1
      int roi_width = max(roi_end_w - roi_start_w + 1, 1);
      int roi_height = max(roi_end_h - roi_start_h + 1, 1);

      Dtype bin_size_h =
          static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w =
          static_cast<Dtype>(roi_width) / static_cast<Dtype>(pooled_width);

      int phstart = floor(((static_cast<Dtype>(h) - 0.5 + 0.5) / spatial_scale -
                           0.5 - roi_start_h) /
                          bin_size_h);
      int phend = ceil(((static_cast<Dtype>(h) + 0.5 + 0.5) / spatial_scale -
                        0.5 - roi_start_h) /
                       bin_size_h);
      int pwstart = floor(((static_cast<Dtype>(w) - 0.5 + 0.5) / spatial_scale -
                           0.5 - roi_start_w) /
                          bin_size_w);
      int pwend = ceil(((static_cast<Dtype>(w) + 0.5 + 0.5) / spatial_scale -
                        0.5 - roi_start_w) /
                       bin_size_w);

      phstart = min(max(phstart, 0), pooled_height);
      phend = min(max(phend, 0), pooled_height);
      pwstart = min(max(pwstart, 0), pooled_width);
      pwend = min(max(pwend, 0), pooled_width);

      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          int pooled_index = ph * pooled_width + pw;
          Dtype max_w = offset_argmax_data_w[pooled_index];
          Dtype max_h = offset_argmax_data_h[pooled_index];

          Dtype sh = (static_cast<Dtype>(max_h) + 0.5) * spatial_scale - 0.5;
          Dtype sw = (static_cast<Dtype>(max_w) + 0.5) * spatial_scale - 0.5;
          int x_left = floor(sw);
          int x_right = ceil(sw);
          int y_top = floor(sh);
          int y_bottom = ceil(sh);

          if (x_left == w && y_top == h)
            gradient += (1 - (sw - -x_left)) * (1 - (sh - y_top)) *
                        offset_top_diff[pooled_index];
          else if (x_right == w && y_top == h)
            gradient += (1 - (x_right - sw)) * (1 - (sh - y_top)) *
                        offset_top_diff[pooled_index];
          else if (x_left == w && y_bottom == h)
            gradient += (1 - (sw - x_left)) * (1 - (y_bottom - sh)) *
                        offset_top_diff[pooled_index];
          else if (x_right == w && y_bottom == h)
            gradient += (1 - (x_right - sw)) * (1 - (y_bottom - sh)) *
                        offset_top_diff[pooled_index];
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}
template <typename Dtype>
__global__ void ROIAlignBackward(const int nthreads, const Dtype* top_diff,
                                 const Dtype* argmax_data_w,
                                 const Dtype* argmax_data_h, const int num_rois,
                                 const Dtype spatial_scale, const int channels,
                                 const int height, const int width,
                                 const int pooled_height,
                                 const int pooled_width, Dtype* bottom_diff,
                                 const Dtype* bottom_rois) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_batch_ind = offset_bottom_rois[0];
      // Skip if ROI's batch index doesn't match n
      if (n != roi_batch_ind) {
        continue;
      }

      Dtype roi_start_w = (offset_bottom_rois[1] + 0.5) * spatial_scale - 0.5;
      Dtype roi_start_h = (offset_bottom_rois[2] + 0.5) * spatial_scale - 0.5;
      Dtype roi_end_w = (offset_bottom_rois[3] + 0.5) * spatial_scale - 0.5;
      Dtype roi_end_h = (offset_bottom_rois[4] + 0.5) * spatial_scale - 0.5;

      // Skip if ROI doesn't include (h, w)
      const bool in_roi = (w >= floor(roi_start_w) && w <= ceil(roi_end_w) &&
                           h >= floor(roi_start_h) && h <= ceil(roi_end_h));
      if (!in_roi) {
        continue;
      }

      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const Dtype* offset_top_diff = top_diff + offset;
      const Dtype* offset_argmax_data_w = argmax_data_w + offset;
      const Dtype* offset_argmax_data_h = argmax_data_h + offset;

      // Compute feasible set of pooled units that could have pooled
      // this bottom unit

      // Force malformed ROIs to be 1x1
      Dtype roi_width = roi_end_w - roi_start_w;
      Dtype roi_height = roi_end_h - roi_start_h;

      Dtype bin_size_h = roi_height / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = roi_width / static_cast<Dtype>(pooled_width);

      int phstart = floor((static_cast<Dtype>(h) - roi_start_h) / bin_size_h);
      int phend =
          ceil((static_cast<Dtype>(h) - roi_start_h + 1.0) / bin_size_h);
      int pwstart = floor((static_cast<Dtype>(w) - roi_start_w) / bin_size_w);
      int pwend =
          ceil((static_cast<Dtype>(w) - roi_start_w + 1.0) / bin_size_w);

      phstart = min(max(phstart, 0), pooled_height);
      phend = min(max(phend, 0), pooled_height);
      pwstart = min(max(pwstart, 0), pooled_width);
      pwend = min(max(pwend, 0), pooled_width);

      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          int pooled_index = ph * pooled_width + pw;
          Dtype max_w = offset_argmax_data_w[pooled_index];
          Dtype max_h = offset_argmax_data_h[pooled_index];

          int x_left = floor(max_w);
          int x_right = ceil(max_w);
          int y_top = floor(max_h);
          int y_bottom = ceil(max_h);

          if (x_left == w && y_top == h)
            gradient += (1 - (max_w - w)) * (1 - (max_h - h)) *
                        offset_top_diff[pooled_index];
          else if (x_right == w && y_top == h)
            gradient += (1 - (w - max_w)) * (1 - (max_h - h)) *
                        offset_top_diff[pooled_index];
          else if (x_left == w && y_bottom == h)
            gradient += (1 - (max_w - w)) * (1 - (h - max_h)) *
                        offset_top_diff[pooled_index];
          else if (x_right == w && y_bottom == h)
            gradient += (1 - (w - max_w)) * (1 - (h - max_h)) *
                        offset_top_diff[pooled_index];
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void ROIAlignLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                        const vector<bool>& propagate_down,
                                        const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  const Dtype* argmax_data_w = max_idx_w_.gpu_data();
  const Dtype* argmax_data_h = max_idx_h_.gpu_data();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ROIAlignBackward<Dtype> << <CAFFE_GET_BLOCKS(count),
                               CAFFE_CUDA_NUM_THREADS>>>
      (count, top_diff, argmax_data_w, argmax_data_h, top[0]->num(),
       spatial_scale_, channels_, height_, width_, pooled_height_,
       pooled_width_, bottom_diff, bottom_rois);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(ROIAlignLayer);

}  // namespace caffe

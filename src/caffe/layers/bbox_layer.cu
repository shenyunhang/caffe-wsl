#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/bbox_layer.hpp"
#include "caffe/util/interp.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/more_math_functions.hpp"

#include <boost/filesystem.hpp>

namespace caffe {

template <typename Dtype>
void Show_bboxes(Blob<Dtype> *bbox_blob, const int num_class_,
                 const int max_bb_per_cls_, const int save_id,
                 const vector<string> voc_label, const bool is_resize,
                 const int height_im_, const int width_im_) {
  cv::RNG rng(12345);

  // load new image
  stringstream load_path;
  load_path << "tmp/" << save_id << "_.png";
  LOG(INFO) << "load png path: " << load_path.str();
  cv::Mat im_mat = cv::imread(load_path.str());
  cv::Mat im_mat_o;

  for (int c = 0; c < num_class_; ++c) {
    const Dtype *bbox_data = bbox_blob->cpu_data() + c * max_bb_per_cls_ * 4;
    if (bbox_data[0] == -1) {
      continue;
    }
    if (is_resize) {
      cv::resize(im_mat, im_mat_o, cv::Size(width_im_, height_im_));
    } else {
      im_mat.copyTo(im_mat_o);
    }
    for (int box_id = 0; box_id < max_bb_per_cls_; ++box_id) {
      if (bbox_data[0] == -1) {
        break;
      }

      cv::rectangle(im_mat_o, cv::Point(bbox_data[0], bbox_data[1]),
                    cv::Point(bbox_data[2], bbox_data[3]),
                    cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
                               rng.uniform(0, 255)),
                    4);

      bbox_data += 4;
    }

    stringstream save_path_o;
    save_path_o << "tmp/" << save_id << "_" << voc_label[c] << "_bbox_o.png";
    cv::imwrite(save_path_o.str(), im_mat_o);
  }
}

template <typename Dtype>
void Show_blob(const Dtype *data, const int channels, const int height,
               const int width, const string save_cpg_path,
               const float threshold_, const int fill = 0) {
  Dtype maxval = -FLT_MAX;
  Dtype sum = 0;
  for (int c = 0; c < channels; c++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        int index = (c * height + h) * width + w;
        /*Dtype value = abs(data[index]);*/
        Dtype value = data[index] > 0 ? data[index] : 0;
        if (value > maxval) {
          maxval = value;
        }
        sum += value;
      }
    }
  }
  LOG(INFO) << "raw mean: " << sum / channels / height / width;

  if (threshold_ > 0) {
    maxval = maxval * threshold_;
  } else {
    maxval = sum / channels / height / width;
  }
  Dtype scale_factor = 255.0 / maxval;

  //-----------------------------------------------------------------------
  cv::Mat cpg_mat;
  if (channels == 3) {
    cpg_mat = cv::Mat(height, width, CV_8UC3);
  } else if (channels == 1) {
    cpg_mat = cv::Mat(height, width, CV_8UC1);
  } else {
    LOG(FATAL) << "channels should 1 or 3";
  }

  sum = 0;
  uchar *cpg_mat_data = cpg_mat.data;
  for (int c = 0; c < channels; c++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        int index = (c * height + h) * width + w;
        int index_mat = (h * width + w) * channels + c;
        /*Dtype value = abs(data[index]);*/
        Dtype value = data[index] > 0 ? data[index] : 0;
        if (value > maxval) {
          cpg_mat_data[index_mat] = 255;
          sum += maxval;
        } else {
          if (fill >= 0) {
            cpg_mat_data[index_mat] = fill;
          } else {
            cpg_mat_data[index_mat] = scale_factor * value;
          }
          sum += value;
        }
      }
    }
  }

  LOG(INFO) << "max_value: " << maxval
            << " mean: " << sum / channels / height / width;

  cv::imwrite(save_cpg_path, cpg_mat);

  //-----------------------------------------------------------------------
  /*const Dtype* cpg_cpu=cpg_blob->cpu_data();*/
  /*int total[26];*/
  /*for(int i=0;i<26;++i){*/
  /*total[i]=0;*/
  /*}*/
  /*for(int e=0;e<cpg_blob->count();e++){*/
  /*int level=int(cpg_cpu[e]/10);*/
  /*total[level]++;*/
  /*}*/
  /*for(int i=0;i<26;++i){*/
  /*std::cout << i<<":"<<total[i]<<" ";*/
  /*}*/
  /*std::cout<<std::endl;*/
  //-----------------------------------------------------------------------
}

template <typename Dtype>
bool BBoxLayer<Dtype>::aou_small(const Dtype *roi, const Dtype bb_offset) {
  // id x1 y1 x2 y2
  const int bbox_num = bboxes_->shape(0);
  for (int b = 0; b < bbox_num; ++b) {
    const Dtype *bbox = bboxes_->cpu_data() + bboxes_->offset(b);
    if (bbox[0] != roi[0]) continue;

    // contain
    if (bbox[1] >= roi[1] - bb_offset && bbox[2] >= roi[2] - bb_offset &&
        bbox[3] <= roi[3] + bb_offset && bbox[4] <= roi[4] + bb_offset)
      return true;

    Dtype ow = std::min(bbox[3], roi[3]) - std::max(bbox[1], roi[1]) + 1;
    Dtype oh = std::min(bbox[4], roi[4]) - std::max(bbox[2], roi[2]) + 1;

    if (ow <= 0 || oh <= 0) continue;

    Dtype ov = ow * oh;
    Dtype s_b = (bbox[3] - bbox[1] + 1) * (bbox[4] - bbox[2] + 1);

    /*float
     * aou=ov/((bbox[2]-bbox[0])*(bbox[3]-bbox[1])+(roi[2]-roi[0])*(roi[3]-roi[1])-ov);*/
    Dtype aou = ov / s_b;
    if (aou > 0.7) return true;
  }

  return false;
}

template <typename Dtype>
Dtype get_num_ob_(const Dtype *in, const int count, const Dtype threshold) {
  Dtype num_ob = 0;
  for (int i = 0; i < count; ++i) {
    if (threshold < (*in)) {
      num_ob++;
    }
    ++in;
  }
  return num_ob;
}

template <typename Dtype>
void BBoxLayer<Dtype>::Score_map_crf() {
  crf_data_->ReshapeLike(*raw_cpg_);
  caffe_copy(crf_data_->count(), raw_cpg_->cpu_data(),
             crf_data_->mutable_cpu_data());

  //-----------------------------------------------------------------------

  const vector<string> layer_names;
  //= net_->layer_names();
  // int conv4_index = -1;
  // int conv5_index = -1;
  for (size_t i = 0; i < layer_names.size(); i++) {
    if (layer_names[i].compare("conv4_3") == 0) {
      // conv4_index = i;
    }
    if (layer_names[i].compare("conv5_3") == 0) {
      // conv5_index = i;
    }
  }

  const vector<int> conv4_bottom_ids;
  //= net_->bottom_ids(conv4_index);
  shared_ptr<Blob<Dtype> > conv4_blob;
  //= net_->blobs()[conv4_bottom_ids[0]];
  const vector<int> conv5_bottom_ids;
  //= net_->bottom_ids(conv5_index);
  shared_ptr<Blob<Dtype> > conv5_blob;
  //= net_->blobs()[conv5_bottom_ids[0]];

  shared_ptr<Blob<Dtype> > feature_blob = conv5_blob;

  //-----------------------------------------------------------------------
  vector<int> crf_cpg_shape = raw_cpg_->shape();
  crf_cpg_shape[1] = 2;
  crf_cpg_->Reshape(crf_cpg_shape);
  caffe_set(crf_cpg_->count(), Dtype(0), crf_cpg_->mutable_cpu_data());

  Blob<Dtype> fusion_blob;
  vector<int> fusion_shape = feature_blob->shape();
  fusion_shape[1] = 1;
  fusion_blob.Reshape(fusion_shape);
  caffe_set(fusion_blob.count(), Dtype(0), fusion_blob.mutable_cpu_data());

  const int a_offset = fusion_blob.offset(0, 1, 0, 0);
  for (int c = 0; c < feature_blob->channels(); ++c) {
    /*caffe_abs(a_offset, feature_blob->cpu_data() + c * a_offset,
     * crf_cpg_->mutable_cpu_diff());*/
    /*caffe_add(a_offset, crf_cpg_->cpu_diff(), crf_cpg_->cpu_data(),
     * crf_cpg_->mutable_cpu_data());*/
    caffe_add(a_offset, feature_blob->cpu_data() + c * a_offset,
              fusion_blob.cpu_data(), fusion_blob.mutable_cpu_data());
  }

  caffe_cpu_interp2<Dtype, false>(
      1, fusion_blob.cpu_data(), 0, 0, fusion_shape[2], fusion_shape[3],
      fusion_shape[2], fusion_shape[3], crf_cpg_->mutable_cpu_data(), 0, 0,
      crf_cpg_shape[2], crf_cpg_shape[3], crf_cpg_shape[2], crf_cpg_shape[3]);

  const Dtype max_value =
      max_element_bbox(crf_cpg_->cpu_data(), crf_cpg_->count());
  const Dtype scale_factor = 1 / (max_value);
  crf_cpg_->scale_data(scale_factor);
  Dtype *crf_cpg = crf_cpg_->mutable_cpu_data();
  for (int i = 0; i < crf_cpg_->count(); ++i) {
    if (crf_cpg[i] < 0.0) {
      crf_cpg[i] = 0;
    }
  }

  if (debug_info_) {
    LOG(INFO) << "max_value: " << (max_value);
  }

  //-----------------------------------------------------------------------
  crf_data_dim_->Reshape(1, 2, 1, 1);
  crf_data_dim_->mutable_cpu_data()[0] = raw_cpg_->shape(2);
  crf_data_dim_->mutable_cpu_data()[1] = raw_cpg_->shape(3);

  //-----------------------------------------------------------------------
  crf_layer_->Forward(crf_bottom_vec_, crf_top_vec_);

  stringstream save_path;
  save_path << "tmp/" << total_im_ << "_feat.png";
  /*Show_blob(crf_output, false, n, rows, cols, channels,
   * save_crf_cpg_path.str());*/
  Show_blob(crf_output_->cpu_data(), 1, crf_output_->height(),
            crf_output_->width(), save_path.str(), 1);

  stringstream save_fusion_path;
  save_fusion_path << "tmp/" << total_im_ << "_fusion.png";
  Show_blob(crf_cpg_->cpu_data(), 1, crf_cpg_->height(), crf_cpg_->width(),
            save_fusion_path.str(), 1);
}

template <typename Dtype>
void BBoxLayer<Dtype>::BBox_crf(const int label) {
  const vector<int> start_bottom_ids;
  //= net_->bottom_ids(start_index_);
  shared_ptr<Blob<Dtype> > im_blob;
  //= net_->blobs()[start_bottom_ids[0]];

  crf_data_->ReshapeLike(*im_blob);
  caffe_copy(crf_data_->count(), im_blob->cpu_data(),
             crf_data_->mutable_cpu_data());

  vector<int> cpg_shape = im_blob->shape();
  cpg_shape[1] = 2;
  crf_cpg_->Reshape(cpg_shape);

  /*caffe_copy(crf_cpg_->count(), im_blob->cpu_diff(),*/
  /*crf_cpg_->mutable_cpu_data());*/

  const int a_offset = crf_cpg_->offset(0, 1, 0, 0);
  caffe_abs(a_offset, im_blob->cpu_diff(), crf_cpg_->mutable_cpu_data());
  caffe_abs(a_offset, im_blob->cpu_diff() + 1 * a_offset,
            crf_cpg_->mutable_cpu_diff());
  caffe_add(a_offset, crf_cpg_->cpu_diff(), crf_cpg_->cpu_data(),
            crf_cpg_->mutable_cpu_data());
  caffe_abs(a_offset, im_blob->cpu_diff() + 2 * a_offset,
            crf_cpg_->mutable_cpu_diff());
  caffe_add(a_offset, crf_cpg_->cpu_diff(), crf_cpg_->cpu_data(),
            crf_cpg_->mutable_cpu_data());

  const Dtype *max_value = std::max_element(
      crf_cpg_->cpu_data(), crf_cpg_->cpu_data() + crf_cpg_->count());

  if (debug_info_) {
    LOG(INFO) << "BBox_crf max_value: " << *max_value;
  }
  const Dtype scale_factor = 1 / (*max_value);
  crf_cpg_->scale_data(scale_factor);
  Dtype *crf_cpg = crf_cpg_->mutable_cpu_data();
  for (int i = 0; i < crf_cpg_->count(); ++i) {
    if (crf_cpg[i] < crf_threshold_) {
      crf_cpg[i] = 0;
    }
  }

  /*caffe_cpu_axpby(a_offset, Dtype(-1), crf_cpg_->cpu_data(), Dtype(0),
   * crf_cpg_->mutable_cpu_data() + a_offset);*/
  /*caffe_add_scalar(a_offset, Dtype(1), crf_cpg_->mutable_cpu_data() +
   * a_offset);*/

  crf_data_dim_->Reshape(1, 2, 1, 1);
  crf_data_dim_->mutable_cpu_data()[0] = im_blob->shape(2);
  crf_data_dim_->mutable_cpu_data()[1] = im_blob->shape(3);

  crf_layer_->Forward(crf_bottom_vec_, crf_top_vec_);

  stringstream save_crf_cpg_path;
  save_crf_cpg_path << "tmp/" << total_im_ << "_" << voc_label_[label]
                    << "_crf.png";
  /*Show_blob(crf_output, false, n, rows, cols, channels,
   * save_crf_cpg_path.str());*/
  Show_blob(crf_output_->cpu_data(), 1, crf_output_->height(),
            crf_output_->width(), save_crf_cpg_path.str(), 1);
}

template <typename Dtype>
__global__ void OR_gpu(const int count, const Dtype *const in_data1,
                       const Dtype *const in_data2, Dtype *const out_data) {
  CUDA_KERNEL_LOOP(index, count) {
    if (in_data1[index] == 1 || in_data2[index] == 1)
      out_data[index] = 1;
    else
      out_data[index] = 0;
  }
}

template <typename Dtype>
bool BBoxLayer<Dtype>::Need_Back(const Dtype label, const Dtype predict) {
  // assum score is betwween 0 ~ 1
  if (this->phase_ == TRAIN && label <= 0.5) return false;
  if (this->phase_ == TRAIN && predict < predict_threshold_) return false;
  if (this->phase_ == TEST && predict < predict_threshold_) return false;

  return true;
}

template <typename Dtype>
void BBoxLayer<Dtype>::After() {
  // this should not in the Reshape function
  // as Reshape function will be call before start
  total_im_ += num_im_;
  accum_im_ += num_im_;

  if (total_im_ % 1280 == 0) {
    LOG(INFO) << "#im: " << total_im_ << " #roi: " << total_roi_
              << " #roi/#im: " << total_roi_ / total_im_;
    LOG(INFO) << "#im: " << accum_im_ << " #roi: " << accum_roi_
              << " #roi/#im: " << accum_roi_ / accum_im_;
    accum_im_ = 0;
    accum_roi_ = 0;
    accum_label_ = 0;
  }
}

template <typename Dtype>
void BBoxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                   const vector<Blob<Dtype> *> &top) {
  if (!is_cpg_) {
    return;
  }

  //-----------------------------------------------------------------------
  // me back
  //-----------------------------------------------------------------------
  const Dtype *predict_data = bottom[bottom_predict_index_]->cpu_data();

  vector<int> gt_class;
  for (int i = 0; i < num_class_; ++i) {
    int index = i;
    if (!Need_Back(predict_data[index], predict_data[index])) continue;
    gt_class.push_back(i);
    LOG_IF(INFO, debug_info_) << "gt class: " << voc_label_[i];
  }
  if (gt_class.size() == 0) {
    After();
    return;
  }

  /*if (is_crf_) Score_map_crf();*/
  for (size_t gt_id = 0; gt_id < gt_class.size(); ++gt_id) {
    int index = gt_class[gt_id];
    LOG_IF(INFO, debug_info_) << "class: " << voc_label_[gt_class[gt_id]]
                              << " score: " << predict_data[index];

    //-----------------------------------------------------------------------
    caffe_gpu_set(raw_cpg_->count(), Dtype(0), raw_cpg_->mutable_gpu_data());
    caffe_gpu_set(raw_cpg_->count(), Dtype(0), raw_cpg_->mutable_gpu_diff());
    for (int channel_id = 0; channel_id < channels_cpg_; ++channel_id) {
      caffe_gpu_abs(cpg_size_, bottom[bottom_cpgs_index_]->gpu_data() +
                                   bottom[bottom_cpgs_index_]
                                       ->offset(gt_id, channel_id, 0, 0),
                    raw_cpg_->mutable_gpu_diff());

      caffe_gpu_add(cpg_size_, raw_cpg_->gpu_data(), raw_cpg_->gpu_diff(),
                    raw_cpg_->mutable_gpu_data());
    }
    /*LOG_IF(INFO,debug_info_)<<"raw_cpg_[0]:"<<raw_cpg_->cpu_data()[0];*/
    const int bbox_num = caffe_cpu_threshold_bbox(
        raw_cpg_, bboxes_, fg_threshold_, gt_class[gt_id]);
    caffe_copy(bboxes_->count(), bboxes_->cpu_data(),
               top[0]->mutable_cpu_data() + top[0]->offset(gt_class[gt_id]));
    total_roi_ += bbox_num;
    //-----------------------------------------------------------------------
    /*if (is_crf_) BBox_crf(cur[0]);*/
  }

  //-----------------------------------------------------------------------
  total_label_ += gt_class.size();
  After();

  //----------------------------------------------------------------------
  // Show patch
  //----------------------------------------------------------------------
  if (debug_info_) {
    bool is_resize = false;
    if (this->phase_ == TEST) is_resize = true;
    Show_bboxes(top[0], num_class_, max_bb_per_cls_, total_im_, voc_label_,
                is_resize, height_im_, width_im_);
  }
}

template <typename Dtype>
void BBoxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
                                    const vector<bool> &propagate_down,
                                    const vector<Blob<Dtype> *> &bottom) {
  for (size_t i = 0; i < bottom.size(); i++) {
    if (propagate_down[i]) {
      caffe_gpu_set(bottom[i]->count(), static_cast<Dtype>(0),
                    bottom[i]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BBoxLayer);

}  // namespace caffe

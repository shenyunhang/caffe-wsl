#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/repartition_layer.hpp"
#include "caffe/util/interp.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/more_math_functions.hpp"

#include <boost/filesystem.hpp>

namespace caffe {

template <typename Dtype>
void Show_rois(Blob<Dtype> *rois_blob, Blob<Dtype> *fliter_blob,
               Blob<Dtype> *label_blob, const int save_id, const int num_im,
               const int height_im, const int width_im, const bool is_resize,
               const vector<string> voc_label, const int ignore_label) {
  const int num_roi = rois_blob->num();
  const int num_class = fliter_blob->channels();
  const Dtype *rois = rois_blob->cpu_data();
  const Dtype *fliter = fliter_blob->cpu_data();
  const Dtype *label = label_blob->cpu_data();

  cv::RNG rng(12345);
  int num_draw = 0;

  // load new image
  stringstream load_path;
  load_path << "tmp/" << save_id << "_.png";
  LOG(INFO) << "load png path: " << load_path.str();
  cv::Mat im_mat = cv::imread(load_path.str());
  cv::Mat im_mat_o;
  cv::Mat im_mat_l;
  cv::Mat im_mat_a;

  int each_page_num = 200;

  for (int c = 0; c < num_class; ++c) {
    if (label[c] <= 0.5) {
      continue;
    }
    if (c == ignore_label) {
      continue;
    }

    int page = -1;

    for (int r = 0; r < num_roi; ++r) {
      if (r % each_page_num == 0) {
        page++;
        if (is_resize) {
          cv::resize(im_mat, im_mat_o, cv::Size(width_im, height_im));
          cv::resize(im_mat, im_mat_l, cv::Size(width_im, height_im));
          cv::resize(im_mat, im_mat_a, cv::Size(width_im, height_im));
        } else {
          im_mat.copyTo(im_mat_o);
          im_mat.copyTo(im_mat_l);
          im_mat.copyTo(im_mat_a);
        }
      }

      // draw rectangle
      cv::rectangle(im_mat_o, cv::Point(rois[5 * r + 1], rois[5 * r + 2]),
                    cv::Point(rois[5 * r + 3], rois[5 * r + 4]),
                    cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
                               rng.uniform(0, 255)),
                    4);

      if (fliter[r * num_class + c] == 1) {
        cv::rectangle(im_mat_l, cv::Point(rois[5 * r + 1], rois[5 * r + 2]),
                      cv::Point(rois[5 * r + 3], rois[5 * r + 4]),
                      cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
                                 rng.uniform(0, 255)),
                      4);
        num_draw++;

      } else {
        cv::rectangle(im_mat_a, cv::Point(rois[5 * r + 1], rois[5 * r + 2]),
                      cv::Point(rois[5 * r + 3], rois[5 * r + 4]),
                      cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
                                 rng.uniform(0, 255)),
                      4);
      }

      if (r % each_page_num == each_page_num - 1) {
        stringstream save_path_o;
        save_path_o << "tmp/" << save_id << "_" << voc_label[c] << "_rois_"
                    << page << "_o.png";
        cv::imwrite(save_path_o.str(), im_mat_o);

        stringstream save_path_l;
        save_path_l << "tmp/" << save_id << "_" << voc_label[c] << "_rois_"
                    << page << "_l.png";
        cv::imwrite(save_path_l.str(), im_mat_l);

        stringstream save_path_a;
        save_path_a << "tmp/" << save_id << "_" << voc_label[c] << "_rois_"
                    << page << "_a.png";
        cv::imwrite(save_path_a.str(), im_mat_a);
      }
    }

    if (num_roi % each_page_num != each_page_num) {
      stringstream save_path_o;
      save_path_o << "tmp/" << save_id << "_" << voc_label[c] << "_rois_"
                  << page << "_o.png";
      cv::imwrite(save_path_o.str(), im_mat_o);

      stringstream save_path_l;
      save_path_l << "tmp/" << save_id << "_" << voc_label[c] << "_rois_"
                  << page << "_l.png";
      cv::imwrite(save_path_l.str(), im_mat_l);

      stringstream save_path_a;
      save_path_a << "tmp/" << save_id << "_" << voc_label[c] << "_rois_"
                  << page << "_a.png";
      cv::imwrite(save_path_a.str(), im_mat_a);
    }
  }

  LOG(INFO) << "num_draw: " << num_draw;
}

template <typename Dtype>
void Show_blob(const Dtype *data, const int channels, const int height,
               const int width, const string save_opg_path,
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
  cv::Mat opg_mat;
  if (channels == 3) {
    opg_mat = cv::Mat(height, width, CV_8UC3);
  } else if (channels == 1) {
    opg_mat = cv::Mat(height, width, CV_8UC1);
  } else {
    LOG(FATAL) << "channels should 1 or 3";
  }

  sum = 0;
  uchar *opg_mat_data = opg_mat.data;
  for (int c = 0; c < channels; c++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        int index = (c * height + h) * width + w;
        int index_mat = (h * width + w) * channels + c;
        /*Dtype value = abs(data[index]);*/
        Dtype value = data[index] > 0 ? data[index] : 0;
        if (value > maxval) {
          opg_mat_data[index_mat] = 255;
          sum += maxval;
        } else {
          if (fill >= 0) {
            opg_mat_data[index_mat] = fill;
          } else {
            opg_mat_data[index_mat] = scale_factor * value;
          }
          sum += value;
        }
      }
    }
  }

  LOG(INFO) << "max_value: " << maxval
            << " mean: " << sum / channels / height / width;

  cv::imwrite(save_opg_path, opg_mat);

  //-----------------------------------------------------------------------
  /*const Dtype* opg_cpu=opg_blob->cpu_data();*/
  /*int total[26];*/
  /*for(int i=0;i<26;++i){*/
  /*total[i]=0;*/
  /*}*/
  /*for(int e=0;e<opg_blob->count();e++){*/
  /*int level=int(opg_cpu[e]/10);*/
  /*total[level]++;*/
  /*}*/
  /*for(int i=0;i<26;++i){*/
  /*std::cout << i<<":"<<total[i]<<" ";*/
  /*}*/
  /*std::cout<<std::endl;*/
  //-----------------------------------------------------------------------
}

template <typename Dtype>
bool RepartitionLayer<Dtype>::aou_small(const Dtype *roi,
                                        const Dtype bb_offset) {
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
void RepartitionLayer<Dtype>::Score_map_crf() {
  crf_data_->ReshapeLike(*raw_opg_);
  caffe_copy(crf_data_->count(), raw_opg_->cpu_data(),
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
  vector<int> crf_opg_shape = raw_opg_->shape();
  crf_opg_shape[1] = 2;
  crf_opg_->Reshape(crf_opg_shape);
  caffe_set(crf_opg_->count(), Dtype(0), crf_opg_->mutable_cpu_data());

  Blob<Dtype> fusion_blob;
  vector<int> fusion_shape = feature_blob->shape();
  fusion_shape[1] = 1;
  fusion_blob.Reshape(fusion_shape);
  caffe_set(fusion_blob.count(), Dtype(0), fusion_blob.mutable_cpu_data());

  const int a_offset = fusion_blob.offset(0, 1, 0, 0);
  for (int c = 0; c < feature_blob->channels(); ++c) {
    /*caffe_abs(a_offset, feature_blob->cpu_data() + c * a_offset,
     * crf_opg_->mutable_cpu_diff());*/
    /*caffe_add(a_offset, crf_opg_->cpu_diff(), crf_opg_->cpu_data(),
     * crf_opg_->mutable_cpu_data());*/
    caffe_add(a_offset, feature_blob->cpu_data() + c * a_offset,
              fusion_blob.cpu_data(), fusion_blob.mutable_cpu_data());
  }

  caffe_cpu_interp2<Dtype, false>(
      1, fusion_blob.cpu_data(), 0, 0, fusion_shape[2], fusion_shape[3],
      fusion_shape[2], fusion_shape[3], crf_opg_->mutable_cpu_data(), 0, 0,
      crf_opg_shape[2], crf_opg_shape[3], crf_opg_shape[2], crf_opg_shape[3]);

  const Dtype max_value = max_element_(crf_opg_->cpu_data(), crf_opg_->count());
  const Dtype scale_factor = 1 / (max_value);
  crf_opg_->scale_data(scale_factor);
  Dtype *crf_opg = crf_opg_->mutable_cpu_data();
  for (int i = 0; i < crf_opg_->count(); ++i) {
    if (crf_opg[i] < 0.0) {
      crf_opg[i] = 0;
    }
  }

  if (debug_info_) {
    LOG(INFO) << "max_value: " << (max_value);
  }

  //-----------------------------------------------------------------------
  crf_data_dim_->Reshape(1, 2, 1, 1);
  crf_data_dim_->mutable_cpu_data()[0] = raw_opg_->shape(2);
  crf_data_dim_->mutable_cpu_data()[1] = raw_opg_->shape(3);

  //-----------------------------------------------------------------------
  crf_layer_->Forward(crf_bottom_vec_, crf_top_vec_);

  stringstream save_path;
  save_path << "tmp/" << total_im_ << "_feat.png";
  /*Show_blob(crf_output, false, n, rows, cols, channels,
   * save_crf_opg_path.str());*/
  Show_blob(crf_output_->cpu_data(), 1, crf_output_->height(),
            crf_output_->width(), save_path.str(), 1);

  stringstream save_fusion_path;
  save_fusion_path << "tmp/" << total_im_ << "_fusion.png";
  Show_blob(crf_opg_->cpu_data(), 1, crf_opg_->height(), crf_opg_->width(),
            save_fusion_path.str(), 1);
}

template <typename Dtype>
void RepartitionLayer<Dtype>::Repartition_crf(const int label) {
  const vector<int> start_bottom_ids;
  //= net_->bottom_ids(start_index_);
  shared_ptr<Blob<Dtype> > im_blob;
  //= net_->blobs()[start_bottom_ids[0]];

  crf_data_->ReshapeLike(*im_blob);
  caffe_copy(crf_data_->count(), im_blob->cpu_data(),
             crf_data_->mutable_cpu_data());

  vector<int> opg_shape = im_blob->shape();
  opg_shape[1] = 2;
  crf_opg_->Reshape(opg_shape);

  /*caffe_copy(crf_opg_->count(), im_blob->cpu_diff(),*/
  /*crf_opg_->mutable_cpu_data());*/

  const int a_offset = crf_opg_->offset(0, 1, 0, 0);
  caffe_abs(a_offset, im_blob->cpu_diff(), crf_opg_->mutable_cpu_data());
  caffe_abs(a_offset, im_blob->cpu_diff() + 1 * a_offset,
            crf_opg_->mutable_cpu_diff());
  caffe_add(a_offset, crf_opg_->cpu_diff(), crf_opg_->cpu_data(),
            crf_opg_->mutable_cpu_data());
  caffe_abs(a_offset, im_blob->cpu_diff() + 2 * a_offset,
            crf_opg_->mutable_cpu_diff());
  caffe_add(a_offset, crf_opg_->cpu_diff(), crf_opg_->cpu_data(),
            crf_opg_->mutable_cpu_data());

  const Dtype *max_value = std::max_element(
      crf_opg_->cpu_data(), crf_opg_->cpu_data() + crf_opg_->count());

  if (debug_info_) {
    LOG(INFO) << "Repartition_crf max_value: " << *max_value;
  }
  const Dtype scale_factor = 1 / (*max_value);
  crf_opg_->scale_data(scale_factor);
  Dtype *crf_opg = crf_opg_->mutable_cpu_data();
  for (int i = 0; i < crf_opg_->count(); ++i) {
    if (crf_opg[i] < crf_threshold_) {
      crf_opg[i] = 0;
    }
  }

  /*caffe_cpu_axpby(a_offset, Dtype(-1), crf_opg_->cpu_data(), Dtype(0),
   * crf_opg_->mutable_cpu_data() + a_offset);*/
  /*caffe_add_scalar(a_offset, Dtype(1), crf_opg_->mutable_cpu_data() +
   * a_offset);*/

  crf_data_dim_->Reshape(1, 2, 1, 1);
  crf_data_dim_->mutable_cpu_data()[0] = im_blob->shape(2);
  crf_data_dim_->mutable_cpu_data()[1] = im_blob->shape(3);

  crf_layer_->Forward(crf_bottom_vec_, crf_top_vec_);

  stringstream save_crf_opg_path;
  save_crf_opg_path << "tmp/" << total_im_ << "_" << voc_label_[label]
                    << "_crf.png";
  /*Show_blob(crf_output, false, n, rows, cols, channels,
   * save_crf_opg_path.str());*/
  Show_blob(crf_output_->cpu_data(), 1, crf_output_->height(),
            crf_output_->width(), save_crf_opg_path.str(), 1);
}

template <typename Dtype>
__global__ void InitFilter_Train(const int count, const Dtype *const label_data,
                                 const int num_class, Dtype *const top_data) {
  CUDA_KERNEL_LOOP(index, count) {
    top_data[index] = 1;
    // const int c = index % num_class;
    // if (label_data[c] > 0.01) {
    //  top_data[index] = 1;
    //} else {
    //  top_data[index] = 0;
    //}
  }
}

template <typename Dtype>
__global__ void InitFilter_Test(const int count, const Dtype *const label_data,
                                const int num_class, Dtype *const top_data) {
  CUDA_KERNEL_LOOP(index, count) {
    const int c = index % num_class;
    if (label_data[c] > 0.01) {
      top_data[index] = 1;
    } else {
      top_data[index] = 0;
    }
  }
}

template <typename Dtype>
void RepartitionLayer<Dtype>::InitFilter(const Dtype *const label_gpu_data,
                                         Dtype *const top_gpu_data) {
  if (is_instance_label_) {
    caffe_gpu_set(num_roi_ * num_class_, Dtype(-1), top_gpu_data);
  } else if (this->phase_ == TRAIN) {
    caffe_gpu_set(num_roi_ * num_class_, Dtype(1), top_gpu_data);
  } else {
    // NOLINT_NEXT_LINE(whitespace/operators)
    InitFilter_Test<Dtype> << <CAFFE_GET_BLOCKS(num_roi_ * num_class_),
                               CAFFE_CUDA_NUM_THREADS>>>
        (num_roi_ * num_class_, label_gpu_data, num_class_, top_gpu_data);
  }
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
__global__ void ScoreBBoxes(const int num_roi, const Dtype *const opg_data,
                            const int num, const int channels, const int height,
                            const int width, const Dtype *const rois_data,
                            const int num_class, const int label,
                            const Dtype threshold, const Dtype min_density,
                            const Dtype all_mass, Dtype *const top_data,
                            const int r = 5) {
  CUDA_KERNEL_LOOP(index, num_roi) {
    const int rois_index = index;

    const Dtype *const roi = rois_data + 5 * rois_index;
    const int wstart = max(int(roi[1]), 0);
    const int hstart = max(int(roi[2]), 0);
    const int wend = min(int(roi[3]), width);
    const int hend = min(int(roi[4]), height);

    /*Dtype sum = 0;*/
    /*Dtype maxval = -FLT_MAX;*/
    Dtype mass = 0;
    for (int c = 0; c < channels; ++c) {
      const Dtype *gradient = opg_data + c * height * width;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          /*sum += gradient[h * width + w];*/
          /*sum += exp(max(r * gradient[h * width + w], kLOG_THRESHOLD));*/
          /*if (maxval < gradient[h * width + w]) {*/
          /*maxval = gradient[h * width + w];*/
          /*}*/
          if (threshold < gradient[h * width + w]) {
            mass++;
          }
        }
      }
      Dtype s = (hend - hstart) * (wend - wstart);
      Dtype dense = 1.0 * weight / s / channels;
      top_data[rois_index * num_class + label] = dense + 1.0 * weight / all_weight;
    }
    Dtype s = (hend - hstart) * (wend - wstart);
    Dtype density = 1.0 * mass / s / channels;
    top_data[rois_index * num_class + label] = density + 1.0 * mass / all_mass;
  }
}

template <typename Dtype>
__global__ void WeightBBoxes(const int num_roi, const Dtype *const opg_data,
                             const int num, const int channels,
                             const int height, const int width,
                             const Dtype *const rois_data, const int num_class,
                             const int cls_id, const Dtype threshold,
                             const Dtype min_density, const Dtype min_mass,
                             Dtype *const top_data, const int r = 5) {
  CUDA_KERNEL_LOOP(index, num_roi) {
    const int rois_index = index;

    const Dtype *const roi = rois_data + 5 * rois_index;
    const int im_index = int(roi[0]);
    const int wstart = max(int(roi[1]), 0);
    const int hstart = max(int(roi[2]), 0);
    const int wend = min(int(roi[3]), width);
    const int hend = min(int(roi[4]), height);

    /*Dtype sum = 0;*/
    /*Dtype maxval = -FLT_MAX;*/
    Dtype mass = 0;
    for (int c = 0; c < channels; ++c) {
      const Dtype *gradient = opg_data + c * height * width;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          /*sum += gradient[h * width + w];*/
          /*sum += exp(max(r * gradient[h * width + w], kLOG_THRESHOLD));*/
          /*if (maxval < gradient[h * width + w]) {*/
          /*maxval = gradient[h * width + w];*/
          /*}*/
          if (threshold < gradient[h * width + w]) {
            mass++;
          }
        }
      }
    }
    Dtype s = (hend - hstart) * (wend - wstart);
    Dtype density = 1.0 * mass / s / channels;
    if (density > min_density && mass > min_mass) {
      top_data[rois_index * num_class + cls_id] = Dtype(1);
    } else {
      top_data[rois_index * num_class + cls_id] = Dtype(0);

      /*if (min_density > 0 && min_mass > 0)*/
      /*top_data[rois_index * num_class + label] =*/
      /*density / min_density + mass / min_mass;*/
      /*else if (min_density > 0)*/
      /*top_data[rois_index * num_class + label] = density / min_density;*/
      /*else*/
      /*top_data[rois_index * num_class + label] = mass / min_mass;*/
    }
    /*top_data[rois_index * num_class + label] =*/
    /*sum / (hend - hstart) / (wend - wstart);*/
    /*top_data[rois_index * num_class + label] = sum;*/
    /*top_data[rois_index * num_class + label] =*/
    /*log(sum / (hend - hstart) / (wend - wstart)) / r;*/
    /*top_data[rois_index * num_class + label] = maxval;*/
  }
}

template <typename Dtype>
__global__ void LabelBBoxes_softmax(
    const int num_roi, const Dtype *const opg_data, const int num,
    const int channels, const int height, const int width,
    const Dtype *const rois_data, const int num_class, const int cls_id,
    const Dtype threshold, const Dtype min_density, const Dtype min_mass,
    Dtype *const top_data, const int r = 5) {
  CUDA_KERNEL_LOOP(index, num_roi) {
    const int rois_index = index;

    const Dtype *const roi = rois_data + 5 * rois_index;
    const int wstart = max(int(roi[1]), 0);
    const int hstart = max(int(roi[2]), 0);
    const int wend = min(int(roi[3]), width);
    const int hend = min(int(roi[4]), height);

    /*Dtype sum = 0;*/
    /*Dtype maxval = -FLT_MAX;*/
    Dtype mass = 0;
    for (int c = 0; c < channels; ++c) {
      const Dtype *gradient = opg_data + c * height * width;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          /*sum += gradient[h * width + w];*/
          /*sum += exp(max(r * gradient[h * width + w], kLOG_THRESHOLD));*/
          /*if (maxval < gradient[h * width + w]) {*/
          /*maxval = gradient[h * width + w];*/
          /*}*/
          if (threshold < gradient[h * width + w]) {
            mass++;
          }
        }
      }
    }
    Dtype s = (hend - hstart) * (wend - wstart);
    Dtype density = 1.0 * mass / s / channels;
    if (density > min_density && mass > min_mass) {
      top_data[rois_index * num_class + cls_id] = Dtype(-1);
    } else {
      top_data[rois_index * num_class + cls_id] = Dtype(num_class - 1);

      /*if (min_density > 0 && min_mass > 0)*/
      /*top_data[rois_index * num_class + label] =*/
      /*density / min_density + mass / min_mass;*/
      /*else if (min_density > 0)*/
      /*top_data[rois_index * num_class + label] = density / min_density;*/
      /*else*/
      /*top_data[rois_index * num_class + label] = mass / min_mass;*/
    }
    /*top_data[rois_index * num_class + label] =*/
    /*sum / (hend - hstart) / (wend - wstart);*/
    /*top_data[rois_index * num_class + label] = sum;*/
    /*top_data[rois_index * num_class + label] =*/
    /*log(sum / (hend - hstart) / (wend - wstart)) / r;*/
    /*top_data[rois_index * num_class + label] = maxval;*/
  }
}

template <typename Dtype>
__global__ void LabelBBoxes(const int num_roi, const Dtype *const opg_data,
                            const int num, const int channels, const int height,
                            const int width, const Dtype *const rois_data,
                            const int num_class, const int cls_id,
                            const Dtype threshold, const Dtype min_density,
                            const Dtype min_mass, Dtype *const top_data,
                            const int r = 5) {
  CUDA_KERNEL_LOOP(index, num_roi) {
    const int rois_index = index;

    const Dtype *const roi = rois_data + 5 * rois_index;
    const int wstart = max(int(roi[1]), 0);
    const int hstart = max(int(roi[2]), 0);
    const int wend = min(int(roi[3]), width);
    const int hend = min(int(roi[4]), height);

    /*Dtype sum = 0;*/
    /*Dtype maxval = -FLT_MAX;*/
    Dtype mass = 0;
    for (int c = 0; c < channels; ++c) {
      const Dtype *gradient = opg_data + c * height * width;
      for (int h = hstart; h < hend; ++h) {
        for (int w = wstart; w < wend; ++w) {
          /*sum += gradient[h * width + w];*/
          /*sum += exp(max(r * gradient[h * width + w], kLOG_THRESHOLD));*/
          /*if (maxval < gradient[h * width + w]) {*/
          /*maxval = gradient[h * width + w];*/
          /*}*/
          if (threshold < gradient[h * width + w]) {
            mass++;
          }
        }
      }
    }
    Dtype s = (hend - hstart) * (wend - wstart);
    Dtype density = 1.0 * mass / s / channels;
    if (density > min_density && mass > min_mass) {
      top_data[rois_index * num_class + cls_id] = Dtype(-1);
    } else {
      top_data[rois_index * num_class + cls_id] = Dtype(0);

      /*if (min_density > 0 && min_mass > 0)*/
      /*top_data[rois_index * num_class + label] =*/
      /*density / min_density + mass / min_mass;*/
      /*else if (min_density > 0)*/
      /*top_data[rois_index * num_class + label] = density / min_density;*/
      /*else*/
      /*top_data[rois_index * num_class + label] = mass / min_mass;*/
    }
    /*top_data[rois_index * num_class + label] =*/
    /*sum / (hend - hstart) / (wend - wstart);*/
    /*top_data[rois_index * num_class + label] = sum;*/
    /*top_data[rois_index * num_class + label] =*/
    /*log(sum / (hend - hstart) / (wend - wstart)) / r;*/
    /*top_data[rois_index * num_class + label] = maxval;*/
  }
}

template <typename Dtype>
bool RepartitionLayer<Dtype>::Need_Repartition(const int cls_id,
                                               const Dtype label,
                                               const Dtype predict) {
  if (cls_id == ignore_label_) return false;
  // assum score is betwween 0 ~ 1
  if (this->phase_ == TRAIN) {
    if (label <= 0.5) return false;
    if (predict > predict_threshold_) {
      return true;
    } else {
      return false;
    }
  } else if (this->phase_ == TEST) {
    if (predict > predict_threshold_) {
      return true;
    } else {
      return false;
    }
  } else {
    LOG(FATAL) << "unkown phase: " << this->phase_;
  }
  LOG(FATAL) << "We should not arrive here!";
  return false;
}

template <typename Dtype>
bool RepartitionLayer<Dtype>::Need_Order(const Dtype label,
                                         const Dtype predict) {
  // assum score is betwween 0 ~ 1
  if (this->phase_ == TRAIN) {
    if (label < 0.5) return false;
    if (is_order_ && predict > predict_order_) {
      return true;
    } else {
      return false;
    }
  } else if (this->phase_ == TEST) {
    return false;
  } else {
    LOG(FATAL) << "unkown phase: " << this->phase_;
  }
  LOG(FATAL) << "We should not arrive here!";
  return false;
}

template <typename Dtype>
void RepartitionLayer<Dtype>::After() {
  // this should not in the Reshape function
  // as Reshape function will be call before start
  total_im_ += num_im_;
  total_roi_ += num_roi_;
  accum_im_ += num_im_;
  accum_roi_ += num_roi_;

  if (total_im_ % 1280 == 0) {
    LOG(INFO) << "#im: " << total_im_ << " #label: " << total_label_
              << " #roi: " << total_roi_ << " #roi_l: " << total_roi_l_
              << " #roi/#num: " << total_roi_ / total_im_
              << " #roi_l/#label: " << total_roi_l_ / total_label_;

    LOG(INFO) << "#im: " << accum_im_ << " #label: " << accum_label_
              << " #roi: " << accum_roi_ << " #roi_l: " << accum_roi_l_
              << " #roi/#num: " << accum_roi_ / accum_im_
              << " #roi_l/#label: " << accum_roi_l_ / accum_label_;

    accum_im_ = 0;
    accum_roi_ = 0;
    accum_roi_l_ = 0;
    accum_label_ = 0;

    if (is_order_) {
      order_threshold_ =
          1.0 - 1.0 * (int(1.0 * total_im_ / order_step_) + 1) / order_K_;
      if (order_threshold_ < 0) order_threshold_ = 0;
      LOG(INFO) << "#im:" << total_im_
                << " order_threshold_: " << order_threshold_;
    }
  }
}

template <typename Dtype>
void RepartitionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                          const vector<Blob<Dtype> *> &top) {
  InitFilter(bottom[bottom_label_index_]->gpu_data(),
             fliter_.mutable_gpu_data());

  if (!is_opg_) {
    top[0]->CopyFrom(fliter_, false, false);
    return;
  }

  //-----------------------------------------------------------------------
  // me back
  //-----------------------------------------------------------------------
  const Dtype *bottom_rois = bottom[bottom_rois_index_]->cpu_data();
  const Dtype *bottom_label = bottom[bottom_label_index_]->cpu_data();
  const Dtype *bottom_predict = bottom[bottom_predict_index_]->cpu_data();
  Dtype *pos_label_data;
  Dtype *neg_label_data;
  if (is_order_ && top.size() == 3) {
    pos_label_data = top[1]->mutable_cpu_data();
    neg_label_data = top[2]->mutable_cpu_data();
  }

  int opg_id = 0;
  int re_num = 0;
  int od_num = 0;
  int gt_num = 0;
  for (int cls_id = 0; cls_id < num_class_; ++cls_id) {
    int index = cls_id;
    LOG_IF(INFO, debug_info_) << "class: " << voc_label_[cls_id]
                              << " label: " << bottom_label[index]
                              << " score: " << bottom_predict[index];

    if (bottom_label[index] > 0.5) ++gt_num;
    if (Need_Repartition(cls_id, bottom_label[index], bottom_predict[index])) {
    } else if (Need_Order(bottom_label[index], bottom_predict[index])) {
    } else {
      continue;
    }

    //-----------------------------------------------------------------------
    caffe_gpu_set(raw_opg_->count(), Dtype(0), raw_opg_->mutable_gpu_data());
    caffe_gpu_set(raw_opg_->count(), Dtype(0), raw_opg_->mutable_gpu_diff());
    for (int channel_id = 0; channel_id < channels_opg_; ++channel_id) {
      caffe_gpu_abs(opg_size_, bottom[bottom_opgs_index_]->gpu_data() +
                                   bottom[bottom_opgs_index_]
                                       ->offset(opg_id, channel_id, 0, 0),
                    raw_opg_->mutable_gpu_diff());
      caffe_gpu_add(opg_size_, raw_opg_->gpu_data(), raw_opg_->gpu_diff(),
                    raw_opg_->mutable_gpu_data());
    }
    ++opg_id;

    // TODO(YH): order_threshold_
    if (order_threshold_ > 0 &&
        Need_Order(bottom_label[index], bottom_predict[index])) {
      ++od_num;
      caffe_cpu_threshold_bbox(raw_opg_, bboxes_, fg_threshold_, cls_id);
      Dtype max_size = 0;
      const Dtype *bbox = bboxes_->cpu_data();
      for (int box_id = 0; box_id < max_bb_per_cls_; ++box_id) {
        if (bbox[0] == -1) break;
        Dtype size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]);
        if (max_size < size) max_size = size;
        bbox += bboxes_->offset(1);
      }
      Dtype max_scale = max_size / (height_im_ * width_im_);
      if (max_scale < order_threshold_) {
        pos_label_data[cls_id] = -1;
        neg_label_data[cls_id] = -1;
        continue;
      }
    }

    if (!Need_Repartition(cls_id, bottom_label[index], bottom_predict[index]))
      continue;
    ++re_num;

    const Dtype maxval = caffe_cpu_max_element(opg_size_, raw_opg_->cpu_data());
    const Dtype threshold = maxval * fg_threshold_;

    const Dtype im_mass =
        get_num_ob_(raw_opg_->cpu_data(), opg_size_, threshold);
    const Dtype im_density = 1.0 * im_mass / height_im_ / width_im_;

    LOG_IF(INFO, debug_info_) << "maxval: " << maxval
                              << " threshold: " << threshold
                              << " im_mass: " << im_mass
                              << " im_density: " << im_density;
    if (is_instance_label_) {
      if (is_instance_softmax_) {
        LOG_IF(INFO, debug_info_) << "LabelBBoxes:";
        // NOLINT_NEXT_LINE(whitespace/operators)
        LabelBBoxes_softmax<Dtype> << <CAFFE_GET_BLOCKS(num_roi_),
                                       CAFFE_CUDA_NUM_THREADS>>>
            (num_roi_, raw_opg_->gpu_data(), 1, 1, height_im_, width_im_,
             bottom_rois, num_class_, cls_id, threshold,
             im_density * density_threshold_, im_mass * mass_threshold_,
             fliter_.mutable_gpu_data());
      } else {
      }
    } else {
      if (is_pred_) {
        LOG_IF(INFO, debug_info_) << "ScoreBBoxes:";
        const Dtype min_density = im_density * density_threshold_;
        // NOLINT_NEXT_LINE(whitespace/operators)
        ScoreBBoxes<Dtype> << <CAFFE_GET_BLOCKS(num_roi_),
                               CAFFE_CUDA_NUM_THREADS>>>
            (num_roi_, raw_opg_->gpu_data(), 1, 1, height_im_, width_im_,
             bottom_rois, num_class_, cls_id, threshold, min_density, im_mass,
             fliter_.mutable_gpu_data());
      } else {
        LOG_IF(INFO, debug_info_) << "WeightBBoxes:";
        // NOLINT_NEXT_LINE(whitespace/operators)
        WeightBBoxes<Dtype> << <CAFFE_GET_BLOCKS(num_roi_),
                                CAFFE_CUDA_NUM_THREADS>>>
            (num_roi_, raw_opg_->gpu_data(), 1, 1, height_im_, width_im_,
             bottom_rois, num_class_, cls_id, threshold,
             im_density * density_threshold_, im_mass * mass_threshold_,
             fliter_.mutable_gpu_data());
      }
    }
  }

  //----------------------------------------------------------------------
  // Show patch
  //----------------------------------------------------------------------
  if (debug_info_) {
    bool is_resize = false;
    if (this->phase_ == TEST) is_resize = true;
    Show_rois(bottom[bottom_rois_index_], &fliter_, bottom[bottom_label_index_],
              total_im_, num_im_, height_im_, width_im_, is_resize, voc_label_,
              ignore_label_);
  }

  if (is_instance_label_) {
    if (is_instance_softmax_) {
      caffe_set(num_roi_, Dtype(num_class_), top[0]->mutable_cpu_data());
      Dtype *instance_label_data = top[0]->mutable_cpu_data();
      const Dtype *fliter_data = fliter_.cpu_data();
      for (int cls_id = 0; cls_id < num_class_; ++cls_id) {
        int index = cls_id;
        if (Need_Repartition(cls_id, bottom_label[index],
                             bottom_predict[index])) {
        } else {
          continue;
        }
        for (int roi_id = 0; roi_id < num_roi_; ++roi_id) {
          Dtype f1 = instance_label_data[roi_id];
          Dtype f2 = fliter_data[num_class_ * roi_id + cls_id];

          if (f1 == num_class_) {
            if (f2 == num_class_) {
              instance_label_data[roi_id] = num_class_;
            } else if (f2 == -1) {
              instance_label_data[roi_id] = -1;
            } else {
              instance_label_data[roi_id] = f2;
            }
          } else if (f1 == -1) {
            if (f2 == num_class_) {
              instance_label_data[roi_id] = -1;
            } else if (f2 == -1) {
              instance_label_data[roi_id] = -1;
            } else {
              instance_label_data[roi_id] = f2;
            }
          } else {
            if (f2 == num_class_) {
            } else if (f2 == -1) {
            } else {
              instance_label_data[roi_id] = -1;
            }
          }
        }
      }
    } else {
      caffe_set(num_roi_, Dtype(-1), top[0]->mutable_cpu_data());
      Dtype *instance_label_data = top[0]->mutable_cpu_data();
      const Dtype *fliter_data = fliter_.cpu_data();
      for (int cls_id = 0; cls_id < num_class_; ++cls_id) {
        int index = cls_id;
        if (Need_Repartition(cls_id, bottom_label[index],
                             bottom_predict[index])) {
        } else {
          continue;
        }
        for (int roi_id = 0; roi_id < num_roi_; ++roi_id) {
          Dtype f1 = instance_label_data[roi_id];
          Dtype f2 = fliter_data[num_class_ * roi_id + cls_id];

          if (f1 == 0) {
            if (f2 == 0) {
            } else if (f2 == -1) {
              instance_label_data[roi_id] = -1;
            } else if (f2 == 1) {
              instance_label_data[roi_id] = 1;
            } else {
              LOG(FATAL) << "we should not be here";
            }
          } else if (f1 == -1) {
            if (f2 == 0) {
            } else if (f2 == -1) {
            } else if (f2 == 1) {
              instance_label_data[roi_id] = 1;
            } else {
              LOG(FATAL) << "we should not be here";
            }
          } else if (f1 == 1) {
            if (f2 == 0) {
            } else if (f2 == -1) {
            } else if (f2 == 1) {
              instance_label_data[roi_id] = -1;
            } else {
              LOG(FATAL) << "we should not be here";
            }
          } else {
            LOG(FATAL) << "we should not be here";
          }
        }
      }
    }
  } else {
    top[0]->CopyFrom(fliter_, false, false);
    if (bottom.size() > bottom_io_index_) {
      if (re_num > 0) {
        int save_id = int(bottom[bottom_io_index_]->cpu_data()[0]);
        LOG_IF(INFO, debug_info_) << "save_id: " << save_id;

        BlobProto save_blob;
        top[0]->ToProto(&save_blob, false);
        stringstream save_path;
        save_path << "data/opg_cache/" << save_id;
        /*LOG(INFO) << "save_path: " << save_path<<" " << save_id;*/
        WriteProtoToBinaryFile(save_blob, save_path.str());
      }

      // NOLINT_NEXT_LINE(whitespace/operators)
      OR_gpu<Dtype> << <CAFFE_GET_BLOCKS(num_roi_ * num_class_),
                        CAFFE_CUDA_NUM_THREADS>>>
          (num_roi_ * num_class_, bottom[bottom_filt_index_]->gpu_data(),
           top[0]->gpu_data(), top[0]->mutable_gpu_data());
    }
  }
  //-----------------------------------------------------------------------
  int roi_left = 0;
  if (is_instance_label_) {
    if (is_instance_softmax_) {
      const Dtype *instance_label_data = top[0]->cpu_data();
      for (int i = 0; i < top[0]->count(); ++i) {
        if (instance_label_data[i] <= num_class_ - 1) roi_left++;
      }
    } else {
      const Dtype *instance_label_data = top[0]->cpu_data();
      for (int i = 0; i < top[0]->count(); ++i) {
        if (instance_label_data[i] == 1) roi_left++;
      }
    }
  } else {
    roi_left = fliter_.asum_data() - (num_class_ - gt_num) * num_roi_;
  }
  total_label_ += gt_num;
  total_roi_l_ += roi_left;
  accum_label_ += gt_num;
  accum_roi_l_ += roi_left;
  After();

  LOG_IF(INFO, debug_info_) << "top: " << top[0]->asum_data();
}

template <typename Dtype>
void RepartitionLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype> *> &top, const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {
  for (size_t i = 0; i < bottom.size(); i++) {
    if (propagate_down[i]) {
      caffe_gpu_set(bottom[i]->count(), static_cast<Dtype>(0),
                    bottom[i]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(RepartitionLayer);

}  // namespace caffe

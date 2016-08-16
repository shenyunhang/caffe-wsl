#include <vector>
#include <time.h>

#include "caffe/layers/opg_layer.hpp"
#include "caffe/util/interp.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string currentDateTime() {
  time_t now = time(0);
  struct tm tstruct;
  char buf[80];
  tstruct = *localtime(&now);
  // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
  // for more information about date/time format
  strftime(buf, sizeof(buf), "%Y-%m-%d", &tstruct);

  return buf;
}

template <typename Dtype>
void OPGLayer<Dtype>::Show_info() {
  if (!is_show_) return;
  is_show_ = false;

  LOG(INFO) << "==============================================";
  LOG(INFO) << "OPG layer:";
  LOG(INFO) << "is_opg: " << is_opg_;
  LOG(INFO) << "debug_info_: " << debug_info_;
  LOG(INFO) << "is_contrast: " << is_contrast_;
  LOG(INFO) << "predict_threshold_:" << predict_threshold_;
  LOG(INFO) << "start_layer_name_: " << start_layer_name_;
  LOG(INFO) << "start_layer_index_: " << start_layer_index_;
  LOG(INFO) << "end_layer_name_: " << end_layer_name_;
  LOG(INFO) << "end_layer_index_: " << end_layer_index_;
  LOG(INFO) << "image_blob_index_: " << image_blob_index_;
  LOG(INFO) << "predict_blob_index_: " << predict_blob_index_;
  for (size_t i = 0; i < opg_blob_name_.size(); ++i) {
    LOG(INFO) << "opg_blob_name_: " << opg_blob_name_[i];
    LOG(INFO) << "opg_blob_index_: " << opg_blob_index_[i];
  }

  const vector<string> layer_names_ = net_->layer_names();
  LOG(INFO) << "----------------------------------------------";
  LOG(INFO) << "show all layer names:";
  for (size_t i = 0; i < layer_names_.size(); i++)
    LOG(INFO) << i << ": " << layer_names_[i];

  const vector<string> blob_names_ = net_->blob_names();
  LOG(INFO) << "----------------------------------------------";
  LOG(INFO) << "show all blob name:";
  for (size_t i = 0; i < blob_names_.size(); i++)
    LOG(INFO) << i << ": " << blob_names_[i];

  LOG(INFO) << "==============================================";
}

template <typename Dtype>
void OPGLayer<Dtype>::Save_param_diff() {
  if (this->phase_ == TEST) return;

  //-----------------------------------------------------------------------
  // Save param
  //-----------------------------------------------------------------------
  /*const vector<shared_ptr<Blob<Dtype> > > params=net_->params();*/
  const vector<shared_ptr<Layer<Dtype> > > layers_ = net_->layers();
  if (!is_history_init_) {
    history_params_.clear();

    /*const vector<string> param_display_names=net_->param_display_names();*/
    /*LOG(INFO) << "----------------------------------------------";*/
    /*LOG(INFO) << "The params to be saved are:";*/
    /*for (int param_id = 0; param_id < params.size(); ++param_id){*/
    /*history_params_.push_back(shared_ptr<Blob<Dtype> >(new
     * Blob<Dtype>(1,1,1,1)));*/
    /*LOG(INFO) << param_id<<": "<<param_display_names[param_id];*/
    /*}*/
    /*LOG(INFO) << "----------------------------------------------";*/

    const vector<string> layer_names = net_->layer_names();
    LOG(INFO) << "----------------------------------------------";
    LOG(INFO) << "The layer params to be saved are:";
    for (int layer_id = start_layer_index_; layer_id <= end_layer_index_;
         ++layer_id) {
      LOG(INFO) << "layer_id: " << layer_id
                << " layer_names: " << layer_names[layer_id];
      const int num_blobs = layers_[layer_id]->blobs().size();
      for (int blob_id = 0; blob_id < num_blobs; ++blob_id) {
        history_params_.push_back(
            shared_ptr<Blob<Dtype> >(new Blob<Dtype>(1, 1, 1, 1)));
        LOG(INFO) << "blob_id: " << blob_id;
      }
    }
    LOG(INFO) << "----------------------------------------------";
  }

  /*for (int param_id = 0; param_id < params.size(); ++param_id){*/
  /*history_params_[param_id]->CopyFrom(*params[param_id],true,true);*/
  /*history_params_[param_id]->CopyFrom(*params[param_id],false,true);*/
  /*}*/

  int p = 0;
  for (int layer_id = start_layer_index_; layer_id <= end_layer_index_;
       ++layer_id) {
    vector<shared_ptr<Blob<Dtype> > > layer_blobs = layers_[layer_id]->blobs();
    const int num_blobs = layer_blobs.size();
    for (int blob_id = 0; blob_id < num_blobs; ++blob_id) {
      history_params_[p]->CopyFrom(*layer_blobs[blob_id], true, true);
      history_params_[p]->CopyFrom(*layer_blobs[blob_id], false, true);
      p++;
    }
  }

  /*is_history_init_=true;*/
  /*return;*/
  //-----------------------------------------------------------------------
  // Save intermediate results
  //-----------------------------------------------------------------------
  const vector<shared_ptr<Blob<Dtype> > > blobs = net_->blobs();
  /*const vector<int> start_bottom_ids=net_->bottom_ids(start_index_);*/
  /*const vector<int> end_top_ids=net_->top_ids(end_index_);*/

  if (!is_history_init_) {
    history_blobs_.clear();
    const vector<string> blob_names = net_->blob_names();
    LOG(INFO) << "----------------------------------------------";
    LOG(INFO) << "The intermediate result blobs to be saved are:";
    /*for(int blob_id = start_bottom_ids[0]; blob_id <= end_top_ids[0];
     * ++blob_id)*/
    for (int blob_id = 0; blob_id < blobs.size(); ++blob_id) {
      history_blobs_.push_back(
          shared_ptr<Blob<Dtype> >(new Blob<Dtype>(1, 1, 1, 1)));
      LOG(INFO) << "blob_id: " << blob_id
                << " blob_names: " << blob_names[blob_id];
    }
    LOG(INFO) << "----------------------------------------------";
  }

  /*for(int blob_id = start_bottom_ids[0]; blob_id <= end_top_ids[0];
   * ++blob_id)*/
  for (int blob_id = 0; blob_id < blobs.size(); ++blob_id) {
    history_blobs_[blob_id]->CopyFrom(*blobs[blob_id], true, true);
    history_blobs_[blob_id]->CopyFrom(*blobs[blob_id], false, true);
  }

  is_history_init_ = true;
}

template <typename Dtype>
void OPGLayer<Dtype>::Restore_param_diff() {
  if (this->phase_ == TEST) return;

  //-----------------------------------------------------------------------
  // Restore param
  //-----------------------------------------------------------------------
  /*const vector<shared_ptr<Blob<Dtype> > > params=net_->params();*/
  /*for (int param_id = 0; param_id < params.size(); ++param_id){*/
  /*[>LOG(INFO) <<"restore param_id: " <<param_id<<"
   * "<<params[param_id]->shape_string()<<"
   * "<<history_params_[param_id]->shape_string();<]*/
  /*params[param_id]->CopyFrom(*history_params_[param_id],true,false);*/
  /*[>LOG(INFO) <<"restore param_id: " <<param_id<<"
   * "<<params[param_id]->shape_string()<<"
   * "<<history_params_[param_id]->shape_string();<]*/
  /*params[param_id]->CopyFrom(*history_params_[param_id],false,false);*/
  /*}*/

  const vector<shared_ptr<Layer<Dtype> > > layers_ = net_->layers();
  int p = 0;
  for (int layer_id = start_layer_index_; layer_id <= end_layer_index_;
       ++layer_id) {
    vector<shared_ptr<Blob<Dtype> > > layer_blobs = layers_[layer_id]->blobs();
    const int num_blobs = layer_blobs.size();
    for (int blob_id = 0; blob_id < num_blobs; ++blob_id) {
      layer_blobs[blob_id]->CopyFrom(*history_params_[p], true, false);
      layer_blobs[blob_id]->CopyFrom(*history_params_[p], false, false);
      p++;
    }
  }

  /*return;*/
  //-----------------------------------------------------------------------
  // Restore intermediate results
  //-----------------------------------------------------------------------
  const vector<shared_ptr<Blob<Dtype> > > blobs = net_->blobs();
  /*const vector<int> start_bottom_ids=net_->bottom_ids(start_index_);*/
  /*const vector<int> end_top_ids=net_->top_ids(end_index_);*/

  /*for(int blob_id = start_bottom_ids[0]; blob_id <= end_top_ids[0];
   * ++blob_id)*/
  for (int blob_id = 0; blob_id < blobs.size(); ++blob_id) {
    /*LOG(INFO) <<"restore blob_id: " <<blob_id<<"
     * "<<blobs[blob_id]->shape_string()<<"
     * "<<history_blobs_[blob_id]->shape_string();*/
    blobs[blob_id]->CopyFrom(*history_blobs_[blob_id], true, false);
    blobs[blob_id]->CopyFrom(*history_blobs_[blob_id], false, false);
  }

  /*net_->ClearParamDiffs();*/
}

template <typename Dtype>
void Show_blob(const Dtype *data, const int channels, const int height,
               const int width, const string save_opg_path,
               const float threshold, const int fill = 0) {
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
  Dtype raw_mean = sum / channels / height / width;
  Dtype raw_maxval = maxval;

  if (threshold > 0) {
    maxval = maxval * threshold;
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

  cv::imwrite(save_opg_path, opg_mat);
  LOG(INFO) << "threshold: " << threshold << "\traw_maxval: " << raw_maxval
            << " raw_mean: " << raw_mean << " max_value: " << maxval
            << " mean: " << sum / channels / height / width;

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
void OPGLayer<Dtype>::Show_opg(const Dtype *opg_data, const int current_label,
                               const string info) {
  LOG(INFO) << "label: " << current_label << " info: " << info;

  //-----------------------------------------------------------------------
  // save bboxes
  /*stringstream load_path;*/
  /*load_path << "tmp/" << save_id_this_ + n << "_.png";*/
  /*cv::Mat im_mat = cv::imread(load_path.str());*/

  /*if (this->phase_ == TEST) {*/
  /*cv::resize(im_mat, im_mat, cv::Size(width_im_, height_im_));*/
  /*}*/

  /*for (int b = 0; b < bboxes_->num(); ++b) {*/
  /*const Dtype *bbox = bboxes_->cpu_data() + bboxes_->offset(b);*/
  /*if (bbox[0] != n) continue;*/
  /*cv::rectangle(im_mat, cv::Point(bbox[1], bbox[2]),*/
  /*cv::Point(bbox[3], bbox[4]), cv::Scalar(0, 0, 255));*/
  /*}*/
  /*stringstream save_bb_path;*/
  /*save_bb_path << "tmp/" << save_id_this_ + n << "_" << voc_label_[label] <<
   * "_bbox.png";*/
  /*cv::imwrite(save_bb_path.str(), im_mat);*/

  //-----------------------------------------------------------------------
  // save OPG
  // string save_subdir = currentDateTime();
  string save_subdir = "";
  stringstream save_opg_path;
  save_opg_path << "tmp/" << save_subdir << "/" << save_id_ << "_"
                << voc_label_[current_label] << "_opg_o" << info << ".png";
  Show_blob(opg_data, channels_opg_, height_im_, width_im_, save_opg_path.str(),
            1, -1);

  stringstream save_opg_path0;
  save_opg_path0 << "tmp/" << save_subdir << "/" << save_id_ << "_"
                 << voc_label_[current_label] << "_opg_0" << info << ".png";
  Show_blob(opg_data, channels_opg_, height_im_, width_im_,
            save_opg_path0.str(), 0);

  stringstream save_opg_path1;
  save_opg_path1 << "tmp/" << save_subdir << "/" << save_id_ << "_"
                 << voc_label_[current_label] << "_opg_fg" << info << ".png";
  Show_blob(opg_data, channels_opg_, height_im_, width_im_,
            save_opg_path1.str(), 0.1);

  for (int t = 0; t < 4; ++t) {
    stringstream save_opg_path2;
    save_opg_path2 << "tmp/" << save_subdir << "/" << save_id_ << "_"
                   << voc_label_[current_label] << "_opg_" << pow(10, -t)
                   << info << ".png";
    Show_blob(opg_data, channels_opg_, height_im_, width_im_,
              save_opg_path2.str(), pow(10, -t));
  }

  //-----------------------------------------------------------------------
}

template <typename Dtype>
void OPGLayer<Dtype>::BackwardDebugInfo(const int layer_id) {
  const vector<vector<Blob<Dtype> *> > bottom_vecs_ = net_->bottom_vecs();
  const vector<string> blob_names_ = net_->blob_names();
  const vector<string> layer_names_ = net_->layer_names();
  const vector<shared_ptr<Layer<Dtype> > > layers_ = net_->layers();
  const vector<vector<bool> > bottom_need_backward_ =
      net_->bottom_need_backward();

  const vector<Blob<Dtype> *> &bottom_vec = bottom_vecs_[layer_id];
  for (int bottom_id = 0; bottom_id < bottom_vec.size(); ++bottom_id) {
    /*if (!bottom_need_backward_[layer_id][bottom_id]) { continue; }*/
    const Blob<Dtype> &blob = *bottom_vec[bottom_id];

    const vector<int> tmp = net_->bottom_ids(layer_id);
    const string &blob_name = blob_names_[tmp[bottom_id]];
    const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG_IF(INFO, Caffe::root_solver()) << "    [Backward] "
                                       << "Layer " << layer_names_[layer_id]
                                       << ", bottom blob " << blob_name
                                       << " diff: " << diff_abs_val_mean;
  }
  for (int param_id = 0; param_id < layers_[layer_id]->blobs().size();
       ++param_id) {
    /*if (!layers_[layer_id]->param_propagate_down(param_id)) { continue; }*/
    const Blob<Dtype> &blob = *layers_[layer_id]->blobs()[param_id];
    const Dtype diff_abs_val_mean = blob.asum_diff() / blob.count();
    LOG_IF(INFO, Caffe::root_solver()) << "    [Backward] "
                                       << "Layer " << layer_names_[layer_id]
                                       << ", param blob " << param_id
                                       << " diff: " << diff_abs_val_mean;
  }
}

template <typename Dtype>
void OPGLayer<Dtype>::Clear_split_diff() {
  for (size_t i = 0; i < split_top_blob_.size(); ++i) {
    caffe_gpu_set(split_top_blob_[i]->count(), static_cast<Dtype>(0),
                  split_top_blob_[i]->mutable_gpu_diff());
  }
}

template <typename Dtype>
bool OPGLayer<Dtype>::Need_Repartition(const Dtype label,
                                               const Dtype predict) {
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
bool OPGLayer<Dtype>::Need_Order(const Dtype label,
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
    return true;
  } else {
    LOG(FATAL) << "unkown phase: " << this->phase_;
  }
  LOG(FATAL) << "We should not arrive here!";
  return false;
}

template <typename Dtype>
void OPGLayer<Dtype>::OPG_back() {
  const vector<shared_ptr<Layer<Dtype> > > layers_ = net_->layers();
  const vector<vector<Blob<Dtype> *> > top_vecs_ = net_->top_vecs();
  const vector<vector<Blob<Dtype> *> > bottom_vecs_ = net_->bottom_vecs();

  if (bottom_need_backward_.size() == 0) {
    bottom_need_backward_ = net_->bottom_need_backward();
    for (int layer_id = end_layer_index_; layer_id >= start_layer_index_;
         --layer_id) {
      std::fill(bottom_need_backward_[layer_id].begin(),
                bottom_need_backward_[layer_id].end(), true);

      vector<bool> this_param_propagate_down;
      for (int j = 0; j < layers_[layer_id]->blobs().size(); j++) {
        this_param_propagate_down.push_back(
            layers_[layer_id]->param_propagate_down(j));
      }
      origin_param_propagate_down_.push_back(this_param_propagate_down);
    }
  }

  for (int layer_id = end_layer_index_; layer_id >= start_layer_index_;
       --layer_id) {
    for (int j = 0; j < layers_[layer_id]->blobs().size(); j++) {
      layers_[layer_id]->set_param_propagate_down(j, false);
    }

    layers_[layer_id]->Backward(top_vecs_[layer_id],
                                bottom_need_backward_[layer_id],
                                bottom_vecs_[layer_id]);

    for (int j = 0; j < layers_[layer_id]->blobs().size(); j++) {
      layers_[layer_id]->set_param_propagate_down(
          j, origin_param_propagate_down_[end_layer_index_ - layer_id][j]);
    }

    if (debug_info_) {
      BackwardDebugInfo(layer_id);
    }
  }
}

template <typename Dtype>
__global__ void Maximum(const int count, Dtype *const data, const int stride,
                        const int all_count) {
  CUDA_KERNEL_LOOP(index, count) {
    for (int i = index; i < all_count; i += stride) {
      if (data[index] < data[i]) {
        data[index] = data[i];
      }
    }
  }
}

template <typename Dtype>
__global__ void Do_threshold(const int count, Dtype *const data,
                             const Dtype thr, const Dtype rep) {
  CUDA_KERNEL_LOOP(index, count) {
    if (data[index] < thr) data[index] = rep;
  }
}

template <typename Dtype>
Dtype max_element_(const Dtype *in, const int count) {
  Dtype max_value = -FLT_MAX;
  for (int i = 0; i < count; ++i) {
    if (max_value < (*in)) {
      max_value = (*in);
    }
    ++in;
  }
  return max_value;
}

template <typename Dtype>
__global__ void tanh_gpu(const int count, Dtype *const data) {
  CUDA_KERNEL_LOOP(index, count) { data[index] = tanh(data[index]); }
}

template <typename Dtype>
void OPGLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                  const vector<Blob<Dtype> *> &top) {
  caffe_gpu_set(top[0]->count(), static_cast<Dtype>(0),
                top[0]->mutable_gpu_data());
  if (!is_opg_) {
    return;
  }

  //-----------------------------------------------------------------------
  // Show info
  //-----------------------------------------------------------------------
  Show_info();

  //-----------------------------------------------------------------------
  // save the history param diff
  //-----------------------------------------------------------------------
  /*Save_param_diff();*/

  //-----------------------------------------------------------------------
  // clear split layer diff
  // When OPG backward, the split layer will sum up the unrelate diff of other
  // stream, so set all diff in split layer to zero.
  Clear_split_diff();

  const Dtype *bottom_label = bottom[bottom_label_index_]->cpu_data();
  const Dtype *predict_data = predict_blob_->cpu_data();

  vector<int> bp_class;
  vector<int> gt_class;
  for (int i = 0; i < num_class_; ++i) {
    int index = i;
    if (Need_Repartition(bottom_label[index], predict_data[index])) {
    } else if (Need_Order(bottom_label[index], predict_data[index])) {
    } else {
      continue;
    }
    bp_class.push_back(i);
    gt_class.push_back(i);
    LOG_IF(INFO, debug_info_) << "gt class: " << voc_label_[i];
  }
  if (gt_class.size() == 0) {
    save_id_ += num_im_;
    return;
  }

  int num_gt = gt_class.size();
  int num_bp;
  if (is_contrast_) {
    num_bp = num_gt > 3 ? num_gt : 3;
  } else {
    num_bp = num_gt;
  }

  while (bp_class.size() < num_bp) {
    Dtype max_score = -FLT_MAX;
    int max_id = -1;
    for (int i = 0; i < num_class_; ++i) {
      if (std::find(bp_class.begin(), bp_class.end(), i) != bp_class.end()) {
        /* v contains x */
        continue;
      } else {
        /* v does not contain x */
        int index = i;
        if (predict_data[index] > max_score) {
          max_id = i;
          max_score = predict_data[index];
        }
      }
    }
    CHECK_NE(max_id, -1) << "max_id can not be -1";
    bp_class.push_back(max_id);
    LOG_IF(INFO, debug_info_) << "addition class: " << voc_label_[max_id];
  }

  // reshape top
  vector<int> top_shape;
  top_shape.push_back(num_gt);
  top_shape.push_back(channels_opg_);
  top_shape.push_back(height_im_);
  top_shape.push_back(width_im_);
  top[0]->Reshape(top_shape);
  caffe_gpu_set(top[0]->count(), Dtype(0), top[0]->mutable_gpu_data());
  caffe_gpu_set(top[0]->count(), Dtype(0), top[0]->mutable_gpu_diff());

  // shape the blob to save opg_blob
  raw_opg_->Reshape(num_bp, opg_blob_.size(), height_im_, width_im_);
  caffe_gpu_set(raw_opg_->count(), Dtype(0), raw_opg_->mutable_gpu_data());
  caffe_gpu_set(raw_opg_->count(), Dtype(0), raw_opg_->mutable_gpu_diff());

  //-----------------------------------------------------------------------
  // me back
  //-----------------------------------------------------------------------
  for (size_t bp_id = 0; bp_id < bp_class.size(); ++bp_id) {
    int index = bp_class[bp_id];
    LOG_IF(INFO, debug_info_) << "class: " << voc_label_[bp_class[bp_id]]
                              << " label: " << bottom_label[index]
                              << " score: " << predict_data[index];

    caffe_gpu_set(predict_blob_->count(), static_cast<Dtype>(0),
                  predict_blob_->mutable_gpu_diff());
    caffe_gpu_set(im_blob_->count(), static_cast<Dtype>(0),
                  im_blob_->mutable_gpu_diff());

    Dtype *predict_diff = predict_blob_->mutable_cpu_diff();
    predict_diff[index] = predict_data[index];

    if (predict_data[index] == 1.0) {
      predict_diff[index] = 0.999;
    } else {
    }

    // if test, we always try find cache first
    if (this->phase_ == TEST && false) {
      bool is_back = false;
      BlobProto cache_proto;
      Blob<Dtype> cache_blob;
      stringstream cache_path;
      for (size_t blob_id = 0; blob_id < opg_blob_.size(); ++blob_id) {
        cache_path.str(string());
        cache_path << "data/opg_cache_test/" << save_id_ << "_"
                   << bp_class[bp_id] << "_" << opg_blob_name_[blob_id];

        if (boost::filesystem::exists(cache_path.str())) {
          ReadProtoFromBinaryFileOrDie(cache_path.str(), &cache_proto);
          cache_blob.FromProto(cache_proto, true);
          caffe_copy(opg_blob_[blob_id]->count(), cache_blob.gpu_data(),
                     opg_blob_[blob_id]->mutable_gpu_diff());
        } else {
          is_back = true;
          break;
        }
      }

      if (is_back) {
        OPG_back();

        for (size_t blob_id = 0; blob_id < opg_blob_.size(); ++blob_id) {
          cache_path.str(string());
          cache_path << "data/opg_cache_test/" << save_id_ << "_"
                     << bp_class[bp_id] << "_" << opg_blob_name_[blob_id];

          cache_blob.ReshapeLike(*opg_blob_[blob_id]);
          caffe_copy(opg_blob_[blob_id]->count(),
                     opg_blob_[blob_id]->gpu_diff(),
                     cache_blob.mutable_gpu_data());
          cache_blob.ToProto(&cache_proto, false);
          WriteProtoToBinaryFile(cache_proto, cache_path.str());
        }
      }
    } else {
      OPG_back();
    }

    // maximum along channel and save to raw_opg_
    for (size_t blob_id = 0; blob_id < opg_blob_.size(); ++blob_id) {
      const int channel_size =
          opg_blob_[blob_id]->shape(2) * opg_blob_[blob_id]->shape(3);
      // NOLINT_NEXT_LINE(whitespace/operators)
      Maximum<Dtype> << <CAFFE_GET_BLOCKS(channel_size),
                         CAFFE_CUDA_NUM_THREADS>>>
          (channel_size, opg_blob_[blob_id]->mutable_gpu_diff(), channel_size,
           opg_blob_[blob_id]->count());

      // copy
      if (is_contrast_) {
        caffe_copy(channel_size, opg_blob_[blob_id]->gpu_diff(),
                   raw_opg_->mutable_gpu_data() +
                       raw_opg_->offset(bp_id, blob_id, 0, 0));
      } else {
        caffe_copy(channel_size, opg_blob_[blob_id]->gpu_diff(),
                   raw_opg_->mutable_gpu_diff() +
                       raw_opg_->offset(bp_id, blob_id, 0, 0));
      }
    }
  }

  if (is_contrast_) {
    for (size_t gt_id = 0; gt_id < gt_class.size(); ++gt_id) {
      for (size_t bp_id = 0; bp_id < bp_class.size(); ++bp_id) {
        if (gt_id == bp_id) continue;
        for (size_t blob_id = 0; blob_id < opg_blob_.size(); ++blob_id) {
          const int channel_size =
              opg_blob_[blob_id]->shape(2) * opg_blob_[blob_id]->shape(3);

          caffe_gpu_sub(
              channel_size,
              raw_opg_->gpu_data() + raw_opg_->offset(gt_id, blob_id, 0, 0),
              raw_opg_->gpu_data() + raw_opg_->offset(bp_id, blob_id, 0, 0),
              raw_opg_->mutable_gpu_diff() +
                  raw_opg_->offset(bp_id, blob_id, 0, 0));

          // NOLINT_NEXT_LINE(whitespace/operators)
          Do_threshold<Dtype> << <CAFFE_GET_BLOCKS(channel_size),
                                  CAFFE_CUDA_NUM_THREADS>>>
              (channel_size, raw_opg_->mutable_gpu_diff() +
                                 raw_opg_->offset(bp_id, blob_id, 0, 0),
               0, 0);

          caffe_gpu_add(
              channel_size,
              raw_opg_->gpu_diff() + raw_opg_->offset(gt_id, blob_id, 0, 0),
              raw_opg_->gpu_diff() + raw_opg_->offset(bp_id, blob_id, 0, 0),
              raw_opg_->mutable_gpu_diff() +
                  raw_opg_->offset(gt_id, blob_id, 0, 0));
        }
      }
    }
  }

  for (size_t gt_id = 0; gt_id < gt_class.size(); ++gt_id) {
    // resize to image size and average
    // from opg_blob_ diff to data
    for (size_t blob_id = 0; blob_id < opg_blob_.size(); ++blob_id) {
      if (opg_blob_[blob_id]->shape(2) == height_im_ &&
          opg_blob_[blob_id]->shape(3) == width_im_) {
        caffe_copy(opg_size_, raw_opg_->gpu_diff() +
                                  raw_opg_->offset(gt_id, blob_id, 0, 0),
                   raw_opg_->mutable_gpu_data() +
                       raw_opg_->offset(gt_id, blob_id, 0, 0));
      } else {
        caffe_gpu_interp2<Dtype, false>(
            1, raw_opg_->gpu_diff() + raw_opg_->offset(gt_id, blob_id, 0, 0), 0,
            0, opg_blob_[blob_id]->shape(2), opg_blob_[blob_id]->shape(3),
            opg_blob_[blob_id]->shape(2), opg_blob_[blob_id]->shape(3),
            raw_opg_->mutable_gpu_data() +
                raw_opg_->offset(gt_id, blob_id, 0, 0),
            0, 0, height_im_, width_im_, height_im_, width_im_);
      }

      Dtype maxval = max_element_(
          raw_opg_->cpu_data() + raw_opg_->offset(gt_id, blob_id, 0, 0),
          opg_size_);

      caffe_gpu_scal(opg_size_, Dtype(1.0 / maxval),
                     raw_opg_->mutable_gpu_data() +
                         raw_opg_->offset(gt_id, blob_id, 0, 0));

      caffe_gpu_add(
          opg_size_, top[0]->gpu_data() + top[0]->offset(gt_id, 0, 0, 0),
          raw_opg_->gpu_data() + raw_opg_->offset(gt_id, blob_id, 0, 0),
          top[0]->mutable_gpu_data() + top[0]->offset(gt_id, 0, 0, 0));

      if (debug_info_) {
        Show_opg(raw_opg_->cpu_data() + raw_opg_->offset(gt_id, blob_id, 0, 0),
                 gt_class[gt_id], "_" + opg_blob_name_[blob_id]);
      }
    }

    if (debug_info_) {
      Show_opg(top[0]->cpu_data() + top[0]->offset(gt_id, 0, 0, 0),
               gt_class[gt_id], "_fusion");
    }

    /*//-----------------------------------------------------------------------*/
  }

  //-----------------------------------------------------------------------
  // restore history param diff
  //-----------------------------------------------------------------------
  /*Restore_param_diff();*/

  save_id_ += num_im_;
}

template <typename Dtype>
void OPGLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype> *> &top,
                                   const vector<bool> &propagate_down,
                                   const vector<Blob<Dtype> *> &bottom) {
  for (size_t i = 0; i < bottom.size(); i++) {
    if (propagate_down[i]) {
      caffe_gpu_set(bottom[i]->count(), static_cast<Dtype>(0),
                    bottom[i]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(OPGLayer);

}  // namespace caffe

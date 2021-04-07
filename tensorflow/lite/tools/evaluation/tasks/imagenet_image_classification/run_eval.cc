/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "tensorflow/lite/tools/evaluation/stages/image_classification_stage.h"
#include "tensorflow/lite/tools/evaluation/tasks/task_executor.h"
#include "tensorflow/lite/tools/evaluation/utils.h"
#include "tensorflow/lite/tools/logging.h"

namespace tflite {
namespace evaluation {

constexpr char kModelFileFlag[] = "model_file";
constexpr char kGroundTruthImagesPathFlag[] = "ground_truth_images_path";
constexpr char kGroundTruthLabelsFlag[] = "ground_truth_labels";
constexpr char kOutputFilePathFlag[] = "output_file_path";
constexpr char kModelOutputLabelsFlag[] = "model_output_labels";
constexpr char kDenylistFilePathFlag[] = "denylist_file_path";
constexpr char kNumImagesFlag[] = "num_images";
constexpr char kInterpreterThreadsFlag[] = "num_interpreter_threads";
constexpr char kDelegateFlag[] = "delegate";
constexpr char kTopk[] = "topk";
constexpr char KDebugMode[] = "debug_mode";

template <typename T>
std::vector<T> GetFirstN(const std::vector<T>& v, int n) {
  if (n >= v.size()) return v;
  std::vector<T> result(v.begin(), v.begin() + n);
  return result;
}

class ImagenetClassification : public TaskExecutor {
 public:
  ImagenetClassification() : num_images_(0), num_interpreter_threads_(1) {}
  ~ImagenetClassification() override {}

 protected:
  std::vector<Flag> GetFlags() final;

  // If the run is successful, the latest metrics will be returned.
  absl::optional<EvaluationStageMetrics> RunImpl() final;

 private:
  void OutputResult(const EvaluationStageMetrics& latest_metrics, std::vector<float>& infer_time) const;
  void OutputResultItem(const EvaluationStageMetrics& latest_metrics, const std::string image_name, const std::string label) const;
  std::string model_file_path_;
  std::string ground_truth_images_path_;
  std::string ground_truth_labels_path_;
  std::string model_output_labels_path_;
  std::string denylist_file_path_;
  std::string output_file_path_;
  std::string delegate_;
  int num_images_;
  int num_interpreter_threads_;
  int topk_{10};
  int debug_mode_{0};
};

std::vector<Flag> ImagenetClassification::GetFlags() {
  std::vector<tflite::Flag> flag_list = {
      tflite::Flag::CreateFlag(kModelFileFlag, &model_file_path_,
                               "Path to test tflite model file."),
      tflite::Flag::CreateFlag(
          kModelOutputLabelsFlag, &model_output_labels_path_,
          "Path to labels that correspond to output of model."
          " E.g. in case of mobilenet, this is the path to label "
          "file where each label is in the same order as the output"
          " of the model."),
      tflite::Flag::CreateFlag(
          kGroundTruthImagesPathFlag, &ground_truth_images_path_,
          "Path to ground truth images. These will be evaluated in "
          "alphabetical order of filename"),
      tflite::Flag::CreateFlag(
          kGroundTruthLabelsFlag, &ground_truth_labels_path_,
          "Path to ground truth labels, corresponding to alphabetical ordering "
          "of ground truth images."),
      tflite::Flag::CreateFlag(
          kDenylistFilePathFlag, &denylist_file_path_,
          "Path to denylist file (optional) where each line is a single "
          "integer that is "
          "equal to index number of denylisted image."),
      tflite::Flag::CreateFlag(kOutputFilePathFlag, &output_file_path_,
                               "File to output metrics proto to."),
      tflite::Flag::CreateFlag(kNumImagesFlag, &num_images_,
                               "Number of examples to evaluate, pass 0 for all "
                               "examples. Default: 0"),
      tflite::Flag::CreateFlag(
          kInterpreterThreadsFlag, &num_interpreter_threads_,
          "Number of interpreter threads to use for inference."),
      tflite::Flag::CreateFlag(
          kDelegateFlag, &delegate_,
          "Delegate to use for inference, if available. "
          "Must be one of {'nnapi', 'gpu', 'hexagon', 'xnnpack'}"),
      tflite::Flag::CreateFlag(
          kTopk, &topk_, "Topk Number"
      ),
      tflite::Flag::CreateFlag(
        KDebugMode, &debug_mode_, "Debug Mode"
      )
  };
  return flag_list;
}

absl::optional<EvaluationStageMetrics> ImagenetClassification::RunImpl() {
  // Process images in filename-sorted order.
  std::string image_md5 = GetMD5(ground_truth_images_path_);
  TFLITE_LOG(INFO) << "load ground_truth_images, checksum: " << md5;
  std::vector<std::string> image_files, ground_truth_image_labels;
  if (GetSortedFileNames(StripTrailingSlashes(ground_truth_images_path_),
                         &image_files) != kTfLiteOk) {
    return absl::nullopt;
  }
  std::string label_md5 = GetMD5(ground_truth_labels_path_);
  TFLITE_LOG(INFO) << "load ground_truth_labels, checksum: " << label_md5;
  if (!ReadFileLines(ground_truth_labels_path_, &ground_truth_image_labels)) {
    TFLITE_LOG(ERROR) << "Could not read ground truth labels file";
    return absl::nullopt;
  }
  if (image_files.size() != ground_truth_image_labels.size()) {
    TFLITE_LOG(ERROR) << "Number of images and ground truth labels is not same, image_files is "
     << image_files.size() << ", ground_truth_image_labels is " << ground_truth_image_labels.size();
    return absl::nullopt;
  }
  std::vector<ImageLabel> image_labels;
  image_labels.reserve(image_files.size());
  for (int i = 0; i < image_files.size(); i++) {
    image_labels.push_back({image_files[i], ground_truth_image_labels[i]});
  }

  // Filter out denylisted/unwanted images.
  if (FilterDenyListedImages(denylist_file_path_, &image_labels) != kTfLiteOk) {
    return absl::nullopt;
  }
  if (num_images_ > 0) {
    image_labels = GetFirstN(image_labels, num_images_);
  }

  std::vector<std::string> model_labels;
  if (!ReadFileLines(model_output_labels_path_, &model_labels)) {
    TFLITE_LOG(ERROR) << "Could not read model output labels file";
    return absl::nullopt;
  }

  EvaluationStageConfig eval_config;
  eval_config.set_name("image_classification");
  auto* classification_params = eval_config.mutable_specification()
                                    ->mutable_image_classification_params();
  auto* inference_params = classification_params->mutable_inference_params();
  inference_params->set_model_file_path(model_file_path_);
  inference_params->set_num_threads(num_interpreter_threads_);
  inference_params->set_delegate(ParseStringToDelegateType(delegate_));
  classification_params->mutable_topk_accuracy_eval_params()->set_k(topk_);

  ImageClassificationStage eval(eval_config);

  eval.SetAllLabels(model_labels);
  if (eval.Init(&delegate_providers_) != kTfLiteOk) return absl::nullopt;

  const int step = image_labels.size() / 100;
  TFLITE_LOG(INFO) << "Test begin";
  std::vector<float> infer_time(image_labels.size());
  for (int i = 0; i < image_labels.size(); ++i) {
    if (debug_mode_ && step > 1 && i % step == 0) {
      TFLITE_LOG(INFO) << "Evaluated: " << i / step << "%";
    }
    eval.SetInputs(image_labels[i].image, image_labels[i].label);
    if (eval.Run() != kTfLiteOk) {
      TFLITE_LOG(INFO) << "sampleid: " << image_labels[i].image << ", result=false";
      return absl::nullopt;
    } else {
      TFLITE_LOG(INFO) << "sampleid: " << image_labels[i].image << ", result=true";
    }
    if (debug_mode_) {
      std::string label = image_labels[i].label.substr(image_labels[i].label.find(" ")+1);
      int in_label = stoi(label);
      in_label++;
      OutputResultItem(eval.LatestMetrics(), image_labels[i].image, model_labels[in_label]);
    }
    const auto& inference_latency = eval.LatestMetrics().process_metrics().image_classification_metrics().inference_latency();
    infer_time.at(i) = inference_latency.last_us() * 0.001;
    TFLITE_LOG(INFO) << "latency_case" << (i+1) << "_latency: " << infer_time.at(i) << " ms";
  }
  const auto latest_metrics = eval.LatestMetrics();
  OutputResult(latest_metrics, infer_time);
  TFLITE_LOG(INFO) << "Test end";
  return absl::make_optional(latest_metrics);
}

void ImagenetClassification::OutputResultItem(
  const EvaluationStageMetrics& latest_metrics, const std::string image_name, const std::string label) const {
  if (!output_file_path_.empty()) {
    std::ofstream metrics_ofile;
    metrics_ofile.open(output_file_path_, std::ios::out);
    metrics_ofile << latest_metrics.SerializeAsString();
    metrics_ofile.close();
  }
  const auto& metrics =
      latest_metrics.process_metrics().image_classification_metrics();
  const auto& accuracy_metrics = metrics.topk_accuracy_metrics();
  int rst = 0;
  static int right = 0;
  int num = latest_metrics.num_runs();
  double diff = (double)(1 + right) / (double)num;
  for (int i = 0; i < accuracy_metrics.topk_accuracies_size(); ++i) {
    if (accuracy_metrics.topk_accuracies(i) >= diff) {
      right++;
      rst = i + 1;
      break;
    }
  }
  if (rst > 0) {
      TFLITE_LOG(INFO) << "sampleid: " << image_name << ", top-k is " << rst << ", result is " << label;
  } else {
      TFLITE_LOG(INFO) << "sampleid: " << image_name << ", predict error";
  }
 }

void ImagenetClassification::OutputResult(
    const EvaluationStageMetrics& latest_metrics, std::vector<float>& infer_time) const {
  if (!output_file_path_.empty()) {
    std::ofstream metrics_ofile;
    metrics_ofile.open(output_file_path_, std::ios::out);
    metrics_ofile << latest_metrics.SerializeAsString();
    metrics_ofile.close();
  }
  const auto& metrics =
        latest_metrics.process_metrics().image_classification_metrics();
  if (debug_mode_) {
    TFLITE_LOG(INFO) << "Num evaluation runs: " << latest_metrics.num_runs();
    const auto& preprocessing_latency = metrics.pre_processing_latency();
    TFLITE_LOG(INFO) << "Preprocessing latency: avg="
                    << preprocessing_latency.avg_us() / 1000.0  << "(ms), std_dev="
                    << preprocessing_latency.std_deviation_us() / 1000.0 << "(ms)";
  }
    const auto& inference_latency = metrics.inference_latency();
  TFLITE_LOG(INFO) << "90th_percentile_latency: "
                   << GetPercentile(infer_time, 90)
                   << "ms, min_latency: " << inference_latency.min_us() * 0.001
                   << "ms, max_latency: " << inference_latency.max_us() * 0.001
                   << "ms";
  const auto& accuracy_metrics = metrics.topk_accuracy_metrics();
  TFLITE_LOG(INFO) << "total_accuracy: ";
  for (int i = 0; i < accuracy_metrics.topk_accuracies_size(); ++i) {
    TFLITE_LOG(INFO) << "Top-" << i + 1
                     << " Accuracy: " << accuracy_metrics.topk_accuracies(i);
  }
}

std::unique_ptr<TaskExecutor> CreateTaskExecutor() {
  return std::unique_ptr<TaskExecutor>(new ImagenetClassification());
}

}  // namespace evaluation
}  // namespace tflite

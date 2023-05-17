#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "ru_gpt_ft/types.hpp"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.h"

struct RuGptFtConfig {
  size_t max_seq_len;
  size_t beam_width;
  uint top_k;
  float top_p;
  float temperature;
  float repetition_penalty;
  float presence_penalty;
  int min_length;
  float shared_contexts_ratio;
  float len_penalty;
  float beam_search_diversity_rate;
  size_t head_num;
  size_t size_per_head;
  size_t vocab_size;
  size_t decoder_layers;
  int request_output_len;
  int start_id;
  int end_id;
};

extern const RuGptFtConfig RU_GPT_FT_DEFAULT_CONFIG;

class RuGptFt final : public IGptInference {
  RuGptFtConfig m_config;
  uint32_t m_random_seed;

  struct cudaDeviceProp m_prop;
  fastertransformer::NcclParam m_tensor_para;
  fastertransformer::NcclParam m_pipeline_para;
  cudaStream_t m_stream;
  cublasHandle_t m_cublas_handle;
  cublasLtHandle_t m_cublaslt_handle;

  std::unique_ptr<std::mutex> m_cublas_wrapper_mutex_ptr{nullptr};
  std::unique_ptr<fastertransformer::Allocator<fastertransformer::AllocatorType::CUDA>>
      m_allocator_ptr{nullptr};
  std::unique_ptr<fastertransformer::cublasAlgoMap> m_cublas_algo_map_ptr{nullptr};
  std::unique_ptr<fastertransformer::cublasMMWrapper> m_cublas_wrapper_ptr{nullptr};
  std::unique_ptr<fastertransformer::ParallelGpt<float>> m_gpt_ptr{nullptr};
  std::unique_ptr<fastertransformer::ParallelGptWeight<float>> m_gpt_weights_ptr{nullptr};

 public:
  explicit RuGptFt(const std::string& model_dir, const RuGptFtConfig& config,
                   const uint32_t random_seed);
  Tokens infer(const Tokens& tokens) override;
};

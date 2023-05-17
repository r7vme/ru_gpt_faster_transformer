#include "ru_gpt_ft/ru_gpt_ft.hpp"

namespace ft = fastertransformer;

const RuGptFtConfig RU_GPT_FT_DEFAULT_CONFIG = {
    .max_seq_len = 2048,
    .beam_width = 1,
    .top_k = 0,
    .top_p = 0.5f,
    .temperature = 1.0f,
    .repetition_penalty = 2.0f,
    .presence_penalty = 0.0f,
    .min_length = 0,
    .shared_contexts_ratio = 1.0f,
    .len_penalty = 0.0f,
    .beam_search_diversity_rate = 0.0f,

    // Model params
    .head_num = 16,
    .size_per_head = 64,
    .vocab_size = 50257,
    .decoder_layers = 24,
    .request_output_len = 1024,
    .start_id = 50256,
    .end_id = 50256,
};

RuGptFt::RuGptFt(const std::string& model_dir, const RuGptFtConfig& config,
                 const uint32_t random_seed)
    : m_config(config), m_random_seed(random_seed) {
  cudaStreamCreate(&m_stream);
  cublasCreate(&m_cublas_handle);
  cublasLtCreate(&m_cublaslt_handle);
  cublasSetStream(m_cublas_handle, m_stream);

  m_cublas_algo_map_ptr = std::make_unique<ft::cublasAlgoMap>();

  m_allocator_ptr = std::make_unique<ft::Allocator<ft::AllocatorType::CUDA>>(ft::getDevice());
  m_cublas_wrapper_mutex_ptr = std::make_unique<std::mutex>();
  m_cublas_wrapper_ptr = std::make_unique<ft::cublasMMWrapper>(
      m_cublas_handle, m_cublaslt_handle, m_stream, m_cublas_algo_map_ptr.get(),
      m_cublas_wrapper_mutex_ptr.get(), m_allocator_ptr.get());
  m_cublas_wrapper_ptr->setFP32GemmConfig();

  ft::check_cuda_error(cudaGetDeviceProperties(&m_prop, 0));

  const size_t hidden_units = config.head_num * config.size_per_head;
  const size_t inter_size = 4 * hidden_units;
  m_gpt_weights_ptr = std::make_unique<ft::ParallelGptWeight<float>>(
      hidden_units, inter_size, config.vocab_size, config.decoder_layers, config.max_seq_len, 1, 0,
      1, 0, 0);
  m_gpt_weights_ptr->loadModel(model_dir);

  ft::AttentionType attention_type =
      ft::getAttentionType<float>(config.size_per_head, ft::getSMVersion(),
                                  true,   // remove_padding
                                  0,      // gpt supports any-seq-length fmha
                                  true,   // is_fuse
                                  false,  // with_relative_position_bias
                                  true);  // causal_mask

  // TODO: use make_unique, for some reason template deduction fails
  m_gpt_ptr = std::unique_ptr<ft::ParallelGpt<float>>(new ft::ParallelGpt<float>(
      0,  // max_batch_size, FT will adjust the buffer automatically.
      0,  // max_seq_len, FT will adjust the buffer automatically.
      0,  // input_length, FT will adjust the buffer automatically.
      config.beam_width, config.head_num, config.size_per_head, inter_size, config.decoder_layers,
      0,   // expert_num
      0,   // moe_k
      {},  // moe_layer_index
      config.vocab_size, config.start_id, config.end_id,
      config.end_id + 1,  // p_prompt_tuning token start id
      ft::PromptLearningType::no_prompt, ft::gptVariantParams{},
      0.0f,         // beam_search_diversity_rate,
      0,            // top_k,
      0.0,          // top_p,
      random_seed,  // random_seed,
      1.0f,         // temperature,
      0.0f,         // len_penalty,
      1.0f,         // repetition_penalty,
      m_tensor_para, m_pipeline_para, m_stream, m_cublas_wrapper_ptr.get(), m_allocator_ptr.get(),
      false, &m_prop, attention_type, false, 0, nullptr, 0, config.shared_contexts_ratio));
}

Tokens RuGptFt::infer(const Tokens& tokens) {
  auto token_num = tokens.size();
  if (token_num == 0) {
    return {};
  }

  int input_length = token_num;
  std::vector<int> input_lengths{input_length};

  int* d_input_ids;
  int* d_input_lengths;
  ft::deviceMalloc(&d_input_ids, input_length, false);
  ft::deviceMalloc(&d_input_lengths, 1, false);
  ft::cudaH2Dcpy(d_input_ids, tokens.data(), input_length);
  ft::cudaH2Dcpy(d_input_lengths, input_lengths.data(), 1);

  const int total_output_len = input_length + m_config.request_output_len;

  int* d_output_ids;
  int* d_sequence_lengths;
  ft::deviceMalloc(&d_output_ids, m_config.beam_width * total_output_len, false);
  ft::deviceMalloc(&d_sequence_lengths, m_config.beam_width, false);
  std::vector<uint32_t> output_seq_len(1, total_output_len);

  std::unordered_map<std::string, ft::Tensor> input_tensors =
      std::unordered_map<std::string, ft::Tensor>{
          {"input_ids", ft::Tensor{ft::MEMORY_GPU, ft::TYPE_INT32,
                                   std::vector<size_t>{1, (size_t)input_length}, d_input_ids}},
          {"input_lengths",
           ft::Tensor{ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{1}, d_input_lengths}},
          {"output_seq_len", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_UINT32, std::vector<size_t>{1},
                                        output_seq_len.data()}},
          {"temperature", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1},
                                     &m_config.temperature}},
          {"len_penalty", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1},
                                     &m_config.len_penalty}},
          {"min_length", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_INT32, std::vector<size_t>{1},
                                    &m_config.min_length}}};

  // TODO: fix this insane float equality comparison that comes from FastTransformer example
  if (m_config.repetition_penalty != 1.0f) {
    input_tensors.insert(
        {"repetition_penalty", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1},
                                          &m_config.repetition_penalty}});
  }
  if (m_config.presence_penalty != 0.0f) {
    input_tensors.insert(
        {"presence_penalty", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1},
                                        &m_config.presence_penalty}});
  }
  if (m_config.top_k == 0 && m_config.top_p == 0.0f) {
    ft::FT_CHECK(m_config.beam_width > 1);
    input_tensors.insert({"beam_search_diversity_rate",
                          ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32, std::vector<size_t>{1},
                                     &m_config.beam_search_diversity_rate}});
  } else {
    input_tensors.insert({"random_seed", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_UINT64,
                                                    std::vector<size_t>{1}, &m_random_seed}});
    if (m_config.top_p != 0.0f) {
      input_tensors.insert({"runtime_top_p", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_FP32,
                                                        std::vector<size_t>{1}, &m_config.top_p}});
    }
    if (m_config.top_k != 0) {
      input_tensors.insert({"runtime_top_k", ft::Tensor{ft::MEMORY_CPU, ft::TYPE_UINT32,
                                                        std::vector<size_t>{1}, &m_config.top_k}});
    }
  }

  std::unordered_map<std::string, ft::Tensor> output_tensors =
      std::unordered_map<std::string, ft::Tensor>{
          {"output_ids",
           ft::Tensor{ft::MEMORY_GPU, ft::TYPE_INT32,
                      std::vector<size_t>{1, m_config.beam_width, (size_t)total_output_len},
                      d_output_ids}},
          {"sequence_length",
           ft::Tensor{ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{1, m_config.beam_width},
                      d_sequence_lengths}}};
  m_gpt_ptr->forward(&output_tensors, &input_tensors, m_gpt_weights_ptr.get());

  // output
  Tokens output;
  size_t output_size = total_output_len * m_config.beam_width;
  output.resize(output_size);
  ft::cudaD2Hcpy(output.data(), d_output_ids, output_size);
  return output;
}

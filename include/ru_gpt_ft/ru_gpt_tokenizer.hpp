#pragma once

#include <unordered_map>

#include "bpe.h"
#include "ru_gpt_ft/types.hpp"

///
/// @brief Tokenizer for cyrillic RU_GPT3
///
/// Based on open source library
/// https://github.com/wangkuiyi/huggingface-tokenizer-in-cxx
///
class RuGptTokenizer final : public IGptTokenizer {
  re2::RE2 m_re;
  BPERanks m_bpe_ranks;
  std::unordered_map<uint8_t, wchar_t> m_b2u;
  std::unordered_map<wchar_t, uint8_t> m_u2b;
  std::unordered_map<std::string, int> m_t2i;
  std::unordered_map<int, std::string> m_i2t;

 public:
  ///
  /// @brief Construct tokenizer from merge.txt and vocab.txt
  ///
  /// To convert vocab.json to vocab.txt use
  /// third_party/huggingface-tokenizer-in-cxx/tool/json-to-txt.py
  ///
  /// @param[in] merges_txt_path path to merge.txt
  /// @param[in] vocab_txt_path path to vocab.txt
  ///
  explicit RuGptTokenizer(const std::string& merges_txt_path, const std::string& vocab_txt_path);

  ///
  /// @brief Tokenize a string. String can consist any unicode characters
  ///
  /// @param[in] s input string
  /// @return array of tokens
  ///
  Tokens tokenize(const std::string& s) override;

  ///
  /// @brief Convert array of tokens to a string
  ///
  /// @param[in] tokens array of tokens
  ///
  std::string detokenize(const Tokens& tokens) override;
};

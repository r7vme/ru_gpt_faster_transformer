#pragma once

#include <unordered_map>

#include "bpe.h"
#include "ru_gpt_ft/types.hpp"

class RuGptTokenizer final : public IGptTokenizer {
  re2::RE2 m_re;
  BPERanks m_bpe_ranks;
  std::unordered_map<uint8_t, wchar_t> m_b2u;
  std::unordered_map<wchar_t, uint8_t> m_u2b;
  std::unordered_map<std::string, int> m_t2i;
  std::unordered_map<int, std::string> m_i2t;

 public:
  explicit RuGptTokenizer(const std::string& merges_txt_path, const std::string& vocab_txt_path);
  Tokens tokenize(const std::string& s) override;
  std::string detokenize(const Tokens& tokens) override;
};

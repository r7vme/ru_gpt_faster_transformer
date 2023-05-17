#include "ru_gpt_ft/ru_gpt_tokenizer.hpp"

const char* REGEX_TOKENIZER =
    "('s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| "
    "?[^\\s\\p{L}\\p{N}]+|\\s+\\(?!\\S\\)|\\s+)";

RuGptTokenizer::RuGptTokenizer(const std::string& merges_txt_path,
                               const std::string& vocab_txt_path)
    : m_re(REGEX_TOKENIZER) {
  bytes_to_unicode(&m_b2u, &m_u2b);

  // load merge.txt
  std::fstream merges(merges_txt_path, std::ios::in);
  load_merge_rules(merges, &m_bpe_ranks);

  // load vocab.txt (converted from vocab.json)
  std::fstream vocab_txt(vocab_txt_path, std::ios::in);
  load_vocab(vocab_txt, &m_t2i, &m_i2t);
}

Tokens RuGptTokenizer::tokenize(const std::string& s) {
  Tokens tokens;
  encode(s, m_re, m_bpe_ranks, m_b2u, m_t2i, &tokens);
  return tokens;
};

std::string RuGptTokenizer::detokenize(const Tokens& tokens) {
  return decode(tokens, m_u2b, m_i2t);
};

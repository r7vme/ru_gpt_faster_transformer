#pragma once

#include <cstdint>
#include <string>
#include <vector>

using Token = int32_t;
using Tokens = std::vector<Token>;

struct IGptInference {
  virtual Tokens infer(const Tokens& tokens) = 0;
  virtual ~IGptInference() {}
};

struct IGptTokenizer {
  virtual Tokens tokenize(const std::string& s) = 0;
  virtual std::string detokenize(const Tokens& tokens) = 0;
  virtual ~IGptTokenizer() {}
};

#include <csignal>
#include <cstdint>
#include <iostream>

#include "ru_gpt_ft/ru_gpt_ft.hpp"
#include "ru_gpt_ft/ru_gpt_tokenizer.hpp"

void signal_handler(int signal_num) { exit(signal_num); }

int main(int argc, char* argv[]) {
  signal(SIGINT, signal_handler);

  uint32_t seed = 0;
  srand(seed);

  RuGptTokenizer t("../examples/ru_gpt_faster_transformer/data/merges.txt",
                   "../examples/ru_gpt_faster_transformer/data/vocab.txt");
  RuGptFt x("../models/ru_gpt/1-gpu/", RU_GPT_FT_DEFAULT_CONFIG, seed);
  /* Tokens input = {28810, 20857, 20068, 8542, 282, 225}; */

  // Александр Сергеевич Пушкин родился в 

  Tokens session_tokens = {};
  while (true) {
    std::string current_string;
    std::getline(std::cin, current_string);
    auto current_tokens = t.tokenize(current_string);

    auto output_tokens = x.infer(current_tokens);

    session_tokens = output_tokens;

    // print 
    std::cout << t.detokenize(output_tokens) << std::endl;
  }

  return 0;
}

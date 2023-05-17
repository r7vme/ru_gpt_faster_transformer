#include <csignal>
#include <cstdint>
#include <iostream>

#include "ru_gpt_ft/ru_gpt_ft.hpp"
#include "ru_gpt_ft/ru_gpt_tokenizer.hpp"

void signal_handler(int signal_num) { exit(signal_num); }

int main(int argc, char* argv[]) {
  signal(SIGINT, signal_handler);

  RuGptTokenizer t("../examples/ru_gpt_faster_transformer/data/merges.txt",
                   "../examples/ru_gpt_faster_transformer/data/vocab.txt");
  RuGptFt x("../models/ru_gpt/1-gpu/", RU_GPT_FT_DEFAULT_CONFIG, 0);

  // Example input: Александр Сергеевич Пушкин родился в
  std::cout << "Русская модель rugpt3medium_based_on_gpt2 запущенная при помощи С++ библиотеки FasterTransformer." << std::endl;
  while (true) {
    std::cout << "Введите текст:" << std::endl;
    std::string input_string;
    std::getline(std::cin, input_string);
    auto current_tokens = t.tokenize(input_string);
    auto output_string = t.detokenize(x.infer(current_tokens));
    std::cout << "RuGPT output:\n" << output_string << std::endl;
  }

  return 0;
}

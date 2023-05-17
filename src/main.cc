#include "ru_gpt_ft/ru_gpt_ft.hpp"

int main(int argc, char* argv[]) {
  uint32_t seed = 0;
  srand(seed);

  RuGptFt x("../models/ru_gpt/1-gpu/", RU_GPT_FT_DEFAULT_CONFIG, seed);
  x.process();

  return 0;
}

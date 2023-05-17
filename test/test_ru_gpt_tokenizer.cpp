#include <catch2/catch_test_macros.hpp>

#include "ru_gpt_ft/ru_gpt_tokenizer.hpp"

TEST_CASE("can tokenize cyrillic string", "[RuGptTokenizer]") {
  // TODO: use CMAKE_CURRENT_SOURCE_DIR
  RuGptTokenizer t("../examples/ru_gpt_faster_transformer/data/merges.txt",
                   "../examples/ru_gpt_faster_transformer/data/vocab.txt");

  Tokens expected_tokens = {28810, 20857, 20068, 8542, 282, 225};
  REQUIRE(t.tokenize("Александр Сергеевич Пушкин родился в ") == expected_tokens);
}

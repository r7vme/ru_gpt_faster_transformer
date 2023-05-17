add_executable(ru_gpt_ft src/main.cc src/ru_gpt_ft.cpp)
target_link_libraries(ru_gpt_ft PUBLIC -lcublas -lcublasLt -lcudart ParallelGpt)
target_include_directories(ru_gpt_ft PRIVATE include)

# originated from https://github.com/wangkuiyi/huggingface-tokenizer-in-cxx
set(TOKENIZER_SRC_DIR "third_party/huggingface-tokenizer-in-cxx")
add_subdirectory("${TOKENIZER_SRC_DIR}/re2")

add_library(tokenizer
  "${TOKENIZER_SRC_DIR}/tokenizer/bpe.cc"
  "${TOKENIZER_SRC_DIR}/tokenizer/bpe.h"
  ${RE2_SOURCES}
  )
target_compile_features(tokenizer PUBLIC cxx_std_11)
target_include_directories(tokenizer PUBLIC ${TOKENIZER_SRC_DIR}/re2)

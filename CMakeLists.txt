# third party dependency
# originated from https://github.com/wangkuiyi/huggingface-tokenizer-in-cxx
set(TOKENIZER_SRC_DIR "third_party/huggingface-tokenizer-in-cxx")
add_subdirectory("${TOKENIZER_SRC_DIR}/re2")
add_library(tokenizer
  ${TOKENIZER_SRC_DIR}/tokenizer/bpe.cc
  ${TOKENIZER_SRC_DIR}/tokenizer/bpe.h
  ${RE2_SOURCES}
  )
target_compile_features(tokenizer PUBLIC cxx_std_11)
target_link_libraries(tokenizer re2)
target_include_directories(tokenizer PUBLIC
  ${TOKENIZER_SRC_DIR}/re2
  ${TOKENIZER_SRC_DIR}/tokenizer
)

add_library(ru_gpt_ft src/ru_gpt_ft.cpp src/ru_gpt_tokenizer.cpp)
target_link_libraries(ru_gpt_ft PUBLIC -lcublas -lcublasLt -lcudart ParallelGpt tokenizer)
target_include_directories(ru_gpt_ft PUBLIC include)

add_executable(ru_gpt_ft_exe src/main.cpp)
target_link_libraries(ru_gpt_ft_exe ru_gpt_ft)

if(BUILD_TESTING)
  # unit tests
  add_subdirectory(third_party/Catch2)
  add_executable(unit_tests
    test/test_ru_gpt_tokenizer.cpp
  )
  target_link_libraries(unit_tests PRIVATE ru_gpt_ft Catch2::Catch2WithMain)
  catch_discover_tests(unit_tests)
endif()

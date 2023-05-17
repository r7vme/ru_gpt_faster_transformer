add_executable(ru_gpt_ft src/main.cc src/ru_gpt_ft.cpp)
target_link_libraries(ru_gpt_ft PUBLIC -lcublas -lcublasLt -lcudart ParallelGpt)
target_include_directories(ru_gpt_ft PRIVATE include)

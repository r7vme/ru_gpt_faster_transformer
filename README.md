# RU_GPT3 inference with FasterTransformer

This repository contains example C++ code to run [RU_GPT3](https://huggingface.co/ai-forever/rugpt3medium_based_on_gpt2)
with [NVIDIA FasterTransformer](https://github.com/NVIDIA/FasterTransformer) library.

Tested on Debian 10 with NVIDIA V100. Inference time for 1024 tokens on a single GPU ~3.6-3.8 seconds.

## Build

Build FasterTransformer library as descibed [here](https://github.com/NVIDIA/FasterTransformer/blob/main/docs/gpt_guide.md#build-the-fastertransformer)

Place this repo under `./examples/ru_gpt_faster_transformer`.

Download [RU_GPT3](https://huggingface.co/ai-forever/rugpt3medium_based_on_gpt2) e.g. to home folder `~/rugpt3medium_based_on_gpt2`.

Convert PyTorch RU_GPT3 to FasterTransformer format by running

```
python ./examples/ru_gpt_faster_transformer/scripts/convert_to_ft.py -i ~/rugpt3medium_based_on_gpt2 -o ./models/ru_gpt -i_g 1
```

This script will create `config.ini` and `.bin` files under `./models/ru_gpt`.

To build `ru_gpt_faster_transformer` add `add_subdirectory(ru_gpt_faster_transformer)` to `xamples/CMakeLists.txt`.

Then build FasterTransformer as usual.

```
cd build
make
```

## Run

```
./bin/ru_gpt_ft_exe
```

Example session

```
Русская модель rugpt3medium_based_on_gpt2 запущенная при помощи С++ библиотеки FasterTransformer.
Введите текст:
Александр Сергеевич Пушкин родился в
RuGPT output:
Александр Сергеевич Пушкин родился в  1799 году. Его отец, Александр Николаевич Пушкин (1773-1826), был участником Отечественной войны 1812 года и Крымской кампании 1853 - 54 годов; его мать – Анна Николаевна Ганнибал родилась 26 де
 абря 1683 г., умерла 27 марта 1800г.).
...more output...
```

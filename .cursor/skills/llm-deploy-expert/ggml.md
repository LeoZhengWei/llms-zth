# B3 GGML + C++

前置：S3+S4。这是**另一条运行时**：C++、算子图、可选量化。**不是** B4 mini-vLLM 的前置，也不是 TRT 的下一档加速。

主线仍是把**我们的** checkpoint 跑起来，禁止改去跑现成 7B GGUF 当完成。

## 顺序

1. `export/export_ggml.py` dump 权重 + 校验和
2. CMake + ggml，读 header（层数/维/词表）
3. 单层 attention 对拍 → 整网 logits 对拍 → greedy
4. 再量化（Q8/Q4）当加课

Tokenizer 第一课可留在 Python，C++ 只吃 token id。

## 文件

`ggml/CMakeLists.txt` `ggml/gpt_model.cpp` `ggml/generate.cpp`（编译器用仓库已有的 cmake/gcc/clang）

## 验收

FP32 last-logits max abs diff < 1e-4；greedy 16 token 与 PyTorch 一致（或记录已知分叉）。CMake 命令入 PROGRESS。

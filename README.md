# llms-zth

亲手做一个小 GPT，搞清生成为何慢，再按瓶颈选运行时。

**主干：** Transformer → 训练 → KV cache 解码 → 测量  
**支线（正交，不是四级加速）：** ONNX｜TensorRT｜GGML/C++｜mini-vLLM

仓库内置部署专家 Agent（会纠正「先转 ONNX / 先写 vLLM」这类顺序）：

- 在本目录开 Cursor Agent 即可
- 课表：`learn/PLAN.md`（一次一课）
- 毕业：`learn/GOALS.md`
- 进度：`learn/PROGRESS.md`
- 技能：`.cursor/skills/llm-deploy-expert/`

模型草稿：`generate_talk/model/build_gpt.py`。

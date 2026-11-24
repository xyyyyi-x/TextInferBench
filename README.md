# TextInferBench · 文本模型推理评测

TextInferBench 是一个个人推理评测项目。用同一个 BGE 中文文本编码器，对照 PyTorch 和 ONNX Runtime 的输出和推理性能。项目实现输入规格控制、动态 ONNX 导出、数值一致性校验、延迟统计及结果绘图。

模型权重来自 [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5/tree/7999e1d3359715c523056ef9478215996d62a620)。项目运行时下载指定版本，并复用本地模型缓存。

## 复跑双后端评测

在项目根目录打开 PowerShell，构建 Linux CPU 镜像并执行默认测试矩阵。

```powershell
New-Item -ItemType Directory -Path results -Force | Out-Null
docker build --target benchmark -t inferbench:cpu .
docker run --rm --mount 'type=volume,source=inferbench-model-cache,target=/model-cache' --mount "type=bind,source=$($PWD.Path)\results,target=/results" inferbench:cpu --output /results/linux-cpu-matrix.json
```

运行时先导出支持动态形状的 ONNX 文件。PyTorch 与 ONNX Runtime 使用同一份权重和分词结果。
默认覆盖两个后端、两档批大小（2、8）和两档 token 位置数（16、64）。每种输入先核对输出形状和数值，再分别预热 10 批、计时 30 批。
计时从调用模型开始，到取得输出结束。模型加载、导出和分词先完成。

报告 JSON 保存测评设备、版本、配置、每次延迟和汇总指标。需要调整测试规格时，可在运行命令末尾加 `--batch-sizes 1 4 --token-lengths 16 32`。模型已缓存时，可在 `docker run` 后加 `--network none -e HF_HUB_OFFLINE=1` 断网复跑。

生成 P95 对照图的环境需要 Matplotlib。

```powershell
python -m pip install matplotlib
python plot_benchmark.py results/linux-cpu-matrix.json --output results/linux-cpu-p95.png
```

- [Windows CPU 验收原始记录](results/windows-cpu-matrix.json)和 [P95 对照图](results/windows-cpu-p95.png)来自同一次运行。每个后端测了四种输入规格，每组计时 30 批。
- [Windows CPU 课堂实测原始记录](results/lesson11-cpu-matrix.json)是独立的一次练习。

Linux CPU 镜像已构建。第 21 课复跑时，容器在导出 ONNX 阶段因内存不足被终止（`OOMKilled=true`，退出码 `137`）。PyTorch 与 ONNX Runtime 的完整对照报告来自 Windows CPU，由本项目的 `benchmark.py` 生成。仓库附有 [Optimum-Benchmark 官方工具的 CPU 示例配置](configs/optimum_cpu.yaml)。

## ONNX Runtime 设备实测

使用在 Windows 导出的同一份动态形状 ONNX 模型，沿用 4 种输入规格，每组预热 10 批、计时 30 批。计时包含从传入主机 NumPy 数组到取回输出的全过程。

- [Linux 容器 CPU 实测](results/linux-ort-cpu-matrix.json)记录了 120 批 ONNX Runtime 推理延迟。将模型导出放在 Windows 完成，容器加载现成的 ONNX 文件进行推理。
- [Windows CPU 与 DirectML 实测](results/windows-ort-directml-matrix.json)记录了 240 批推理延迟，四组输出均通过数值校验，最大绝对差约为 6.91 × 10⁻⁶。批大小 8、序列长度 64 时，P95 从 CPU 的 125.25 ms 降至 DirectML 的 20.12 ms。算子记录确认矩阵乘法在 RTX 4060 Laptop GPU 的 DirectML 后端执行。
- [Linux Docker CPU 与 CUDA 实测](results/linux-ort-cuda-matrix.json)使用 ONNX Runtime 1.26.0 记录了 240 批推理延迟，四组输出均通过数值校验，最大绝对差约为 6.20 × 10⁻⁶。批大小 8、序列长度 64 时，P95 为 CPU 161.95 ms、CUDA 3.70 ms。算子记录确认矩阵乘法在 RTX 4060 Laptop GPU 的 CUDA 后端执行。本次实测将两个后端分别放在独立容器中顺序运行，并为 WSL 临时增加 2 GiB 交换空间。

## BERT 前向推理示例

用 `BAAI/bge-small-zh-v1.5` 在 CPU 上把中文句子编码成向量。运行一次，观察文本怎样变成 token 编号、注意力掩码和输出张量。

### 在 PowerShell 运行

打开 Docker Desktop，确认它正在运行 Linux 容器。在项目根目录的 PowerShell 执行下列命令。

```powershell
docker build -t inferbench-lesson1:cpu .
docker run --rm --mount 'type=volume,source=inferbench-model-cache,target=/model-cache' inferbench-lesson1:cpu
```

构建镜像时会安装 CPU 版 PyTorch 和 Transformers。首次运行会联网下载固定版本 `7999e1d3359715c523056ef9478215996d62a620` 的模型文件。后续运行可以通过同一个 Docker 卷复用缓存。

模型已缓存时，可以断网复跑。

```powershell
docker run --rm --network none -e HF_HUB_OFFLINE=1 --mount 'type=volume,source=inferbench-model-cache,target=/model-cache' inferbench-lesson1:cpu
```

### 看懂默认输出

- `tokens` 展示每条句子分词后的标记。`input_ids` 是这些标记的编号，预期形状是 `(2, 16)`。
- `attention_mask` 的预期形状也是 `(2, 16)`。`1` 标记可被关注的位置，`0` 让模型在注意力计算中忽略补齐项。对比两条默认句子，找出 `[PAD]` 和 `0` 的对应关系。
- `last_hidden_state` 的预期形状是 `(2, 16, 512)`。`2` 是一次输入的句子数，即批大小。`16` 是每条输入的 token 位置数，包含 `[CLS]`、`[SEP]` 和补齐位置。`512` 是模型为每个位置输出的向量维度。脚本会检查这个形状。

### 动手练习

先猜把输入改为一条句子、最大长度改为 12 后，输入和输出的形状会怎样变化，再运行验证。

```powershell
docker run --rm --mount 'type=volume,source=inferbench-model-cache,target=/model-cache' inferbench-lesson1:cpu --max-length 12 '我想学习模型推理。'
```

打开 `lesson01_forward.py`，沿着 `tokenizer(...)`、`model(**model_inputs)` 和 `.last_hidden_state` 找到分词、前向和取输出的三步。编辑这个脚本后，可以挂载当前目录来运行新版本，省去再次构建镜像。

```powershell
docker run --rm --mount 'type=volume,source=inferbench-model-cache,target=/model-cache' --mount "type=bind,source=$($PWD.Path),target=/app,readonly" inferbench-lesson1:cpu
```

# efficientdet-runner
Minimal code to run the official EfficientDet model.

本项目提供了执行EfficientDet模型推演的最小代码模块。

## 准备工作
最小代码意味着你需要以 `SavedModel` 格式存储模型。请在官方实现中完成模型转换工作。

### 获取官方源代码
```bash
git clone https://github.com/google/automl.git
```

### 下载模型Checkpoint
EfficientDet包含多个不同规模的实现。这里以`D0`为例，模型名称为 `efficientdet-d0`。从官网下载checkpoin文件。

```bash
wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco/efficientdet-d0.tar.gz
tar xf efficientdet-d0.tar.gz
```

模型checkpoint被解压缩到 `efficientdet-d0` 文件夹。

### 转换Checkpoint以支持TensorFlow 2
使用 `keras` 目录中的 `inspector.py`，将checkpoint转换为TensorFlow 2支持的格式，并存储到 `ckpt-tf2` 文件夹。

```bash
python3 -m keras.inspector --mode=dry --model_name=efficientdet-d0 --model_dir=efficientdet-d0 --export_ckpt=ckpt-tf2/efficientdet-d0
```

### 导出为 `SavedModel` 格式
以推演为主要目的，`SavedModel` 格式可以在没有源代码的情况下执行，更加方便。

```bash
python3 -m keras.inspector --mode=export --model_name=efficientdet-d0 \
  --model_dir=ckpt-tf2 --saved_model_dir=saved_model
```

该命令会同时保存一份冻结后的模型文件，以便有需要的情况下使用。

## 运行
获取本项目的代码。

```bash
git clone https://github.com/yinguobing/efficientdet-runner.git
```

将导出的模型文件夹存储在 `saved_model` 目录下。

检测一段视频。

```bash
python3 demo.py --video=input.mov
```

## 致谢
官方实现：https://github.com/google/automl
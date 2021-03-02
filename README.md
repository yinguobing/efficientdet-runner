# efficientdet-runner
Minimal code to run the EfficientDet model.

## 准备工作
官方模型默认使用TensorFlow 1。但是提供了Keras实现来支持TensorFlow 2。具体可参考`keras`目录。

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
检测一段视频。

```bash
python3 runner.py --video=input.mov
```

## 致谢
官方实现：https://github.com/google/automl
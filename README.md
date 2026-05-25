# Battery SOC Estimation

基于深度学习的电池荷电状态（State of Charge, SOC）估计框架。支持多种时序编码器（LSTM / GRU / CNN / TCN / FCN）、插拔式池化层和回归头，通过 YAML 配置驱动实验。

## 工程架构

```
configs/                  ← YAML 配置文件（继承合并）
  base/default.yaml       基础配置
  experiments/*.yaml      实验特定配置
scripts/
  prepare_data.py         离线数据转换（Excel → Canonical CSV）
  train.py                训练入口
  eval.py                 评估入口
src/
  data/                   数据处理层
    converters/            离线数据转换器（自有设备 Excel）
    dataset.py             DataLoader 构建 + 数据集划分
    window.py              滑动窗口生成
    preprocess.py          Z-score 标准化器
    schema.py              规范化数据契约校验
    io.py                  CSV 加载 + manifest 读取
  models/                 模型层
    encoders/              LSTM / GRU / FCN / CNN / TCN 编码器
    pooling/               Last / Mean / Max / Attention 池化
    head.py                RegressionHead（线性或隐藏层+ReLU）
    soc_model.py           Encoder → Pooling → Head 组合模型
    registry.py            组件注册表（编码器/池化/头部/模型）
  training/               训练层
    trainer.py             Trainer（早停 + 检查点）
    losses.py              MSE / MAE / Smooth L1 注册表
    optimizers.py          Adam / AdamW / SGD 注册表
    checkpoint.py          模型保存与加载
  evaluation/             评估层
    metrics.py             MSE / MAE / RMSE / Max Error / R²
    plots.py               训练曲线 / 预测对比 / 逐序列 SOC 图
  utils/                  工具层
    config.py              YAML 加载 + extends 继承 + 深度合并
    plugins.py             插件动态导入
    seed.py                全局随机种子
    logger.py              统一日志
  experiment.py            实验管理（设备选择、评估保存）
tests/                    单元测试
outputs/experiments/      实验结果输出
```

### 架构设计原则

**分层解耦**：数据处理、模型、训练、评估四层独立，通过配置字典（`config: dict`）在 `scripts/train.py` 中串联。

**注册表模式**：模型组件（编码器、池化、头部、损失、优化器）全部通过注册表 + 字符串名称构建。添加新组件只需 `register_*` 并在配置中指定名称，无需修改训练流水线。

```python
# 示例：注册并使用自定义编码器
from src.models import register_encoder
register_encoder("transformer", build_transformer)
# 然后在 YAML 中写 model.name: transformer 即可
```

**配置驱动**：所有超参数通过 YAML 控制，支持 `extends` 链式继承和 `deep_merge`。基础配置定义默认值，实验配置只覆写差异部分。

**插件扩展**：通过 `config.plugins` 声明外部模块路径，利用 Python 导入副作用自动注册自定义组件，无需侵入现有代码。

**归一化防泄露**：`Standardizer` 仅在训练集上拟合（`fit(train_values)`），然后整体变换，避免验证/测试集信息泄露。

**划分防泄露**：默认按 `sequence_id` 维度依据 `data.split` 比例随机划分，确保同一电池循环的所有窗口落在同一集合中；也可通过 `data.split_column` 使用数据中已经定义好的划分。

## 数据格式

### 规范化 CSV 格式（Canonical CSV）

所有模型训练共用同一数据契约，包含以下核心列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `sequence_id` | str | 序列标识（如电池循环 ID） |
| `time` | float | 时间戳（秒），每个序列内单调递增 |
| `soc` | float | 荷电状态目标值，范围 [0, 1] |
| 特征列（可配置） | float | 如 voltage、current、temperature 等 |

特征列由配置中的 `data.feature_columns` 指定，可自由增减，不修改任何流水线代码。

### 离线数据源

目前支持将**自有电池循环仪导出的 Excel 工作簿**转换为 Canonical CSV：

| 转换器 | 输入 | SOC 计算方法 |
|--------|------|-------------|
| `cycler_workbook` | 循环仪 Excel（含 record + auxTemp 工作表） | 安时积分 + 充/放电容量比 |

转换逻辑位于 `src/data/converters/cycler_workbook.py`，包括：
1. 1 Hz 去重采样
2. 电流符号归一化（充电为正、放电为负）
3. 安时积分（Ah 累计）
4. 温度映射（从 auxTemp 工作表中匹配）
5. 基于充/放电容量的 SOC 计算
6. 列名映射、manifest 与原始文件指纹生成

## 完整训练流程

### 1. 配置加载

```
configs/base/default.yaml  ←  extends 继承链解析
        ↓ deep_merge
configs/experiments/xxx.yaml
        ↓ load_config()
完整配置字典
```

`load_config()` 递归解析 `extends` 链，然后将实验配置深度合并到基础配置上，生成最终运行配置。

### 2. 随机种子与插件

```python
seed_everything(config["seed"])      # Python / NumPy / PyTorch
load_plugins(config)                  # 导入外部插件模块
```

### 3. 数据准备（离线转换）

若配置中包含 `data.raw_path`，训练脚本会自动判断是否需要调用 `prepare_cycler_workbooks()` 将原始 Excel 转为规范化 CSV。训练前会根据 `manifest` 检查 CSV 的序列数、总行数和列结构，并检查原始文件集合及内容指纹；只有产物完整且数据源未变化时才直接复用。旧版完整产物首次复用时会自动补写原始文件指纹，不重新转换 CSV。不完整或过期的产物会自动重新处理，重建时会清除该数据集目录中已失效的旧序列 CSV。相同输入在单次多实验命令中只处理一次。

### 4. 数据加载与预处理

```
load_canonical_csv(root, data_config)
  → glob 匹配 CSV → pd.concat → validate_canonical_frame()
    （检查必需列、数值合法性、SOC ∈ [0,1]、时间单调性）
        ↓
_assign_splits(frame)
  → 默认按 sequence_id 与 data.split 比例随机划分 train/val/test
  → 若提供 data.split_column，则使用 CSV 中已有划分
        ↓
Standardizer.fit(train_values)     ← 仅在训练集上拟合 Z-score
Standardizer.transform(all_values)
        ↓
build_windows(subset, window_size, stride)
  → 每个 sequence 内独立生成滑动窗口，不跨边界
  → 窗口标签 = 窗口最后时间步的 SOC
  → (n_windows, window_size, n_features) 特征 + (n_windows,) 目标
        ↓
SOCDataset(windowed) → DataLoader
  train: shuffle=True
  val/test: shuffle=False
        ↓
DataBundle(loaders, datasets, artifacts, input_dim)
```

### 5. 模型构建

```python
build_model(config["model"], input_dim)
  → model.architecture.name = "encoder_pooling_head"  ← 默认架构
    → build_encoder(config, input_dim)     # 如 LSTM(hidden=64, layers=2)
    → build_pooling(pooling_name, dim)     # 如 LastPooling
    → build_head(head_config, dim)         # 如 RegressionHead
    → SOCModel(encoder, pooling, head)
```

模型前向：`(batch, seq_len, n_features) → encoder → (batch, seq_len, hidden) → pooling → (batch, hidden) → head → (batch, 1)`

### 6. 训练循环

```python
Trainer(model, criterion, optimizer, device, patience, min_delta)
  → for epoch in 1..epochs:
      _epoch(train_loader, training=True)
        → 逐 batch 前向 → loss.backward → optimizer.step
        → 加权平均损失 = Σ(loss × batch_size) / total_samples
      _epoch(val_loader, training=False)
        → 同计算，无梯度
      if val_loss < best_val_loss - min_delta:
        save_checkpoint(..., model_state, optimizer_state, epoch, ...)
        best_epoch ← epoch, stale_epochs ← 0
      else:
        stale_epochs += 1
        if stale_epochs ≥ patience → 早停退出
  → TrainingResult(history, best_epoch, best_val_loss, checkpoint_path)
```

### 7. 评估与输出

```
load_checkpoint(best.pt) → model.load_state_dict
        ↓
predict(model, test_loader, device)
  → 批量前向推理 → (actual, predicted, indices)
        ↓
evaluate_and_save(model, bundle, device, output_dir)
  → predictions.csv      (sequence_id, time, actual_soc, predicted_soc, error)
  → metrics.json         (mse, mae, rmse, max_error, r2)
  → plots/
      soc_prediction_curve.png   实际 vs 预测 SOC 曲线
      pred_vs_true.png           散点图 + 对角线
      training_curve.png         训练/验证损失曲线
      soc_over_time_by_sequence/
        {id}_soc_over_time.png  每个测试序列的逐时间步 SOC 图
```

## 快速开始

### 环境

```bash
pip install -r requirements.txt
```

主要依赖：`torch`, `numpy`, `pandas`, `PyYAML`, `matplotlib`, `openpyxl`, `tqdm`。

### 准备数据

```bash
# 将自有循环仪 Excel 工作簿转为规范化 CSV
python scripts/prepare_data.py \
  --input "data/raw/0.?C.xlsx" \
  --output data/processed/own_cell \
  --overwrite
```

### 训练

```bash
# 单实验
python scripts/train.py --configs configs/experiments/temp.yaml

# 多实验（glob 模式）
python scripts/train.py --configs configs/experiments/*.yaml

# 指定多个配置
python scripts/train.py --configs configs/experiments/exp1.yaml configs/experiments/exp2.yaml
```

### 评估

以下命令假设已经完成 `temp.yaml` 的训练并生成最佳检查点。

```bash
# 重新评估训练时保存的原测试集
python scripts/eval.py \
  --checkpoint outputs/experiments/temp/checkpoints/best.pt

# 使用已训练模型评估新的 Excel 数据集，不重新训练
python scripts/eval.py \
  --checkpoint outputs/experiments/temp/checkpoints/best.pt \
  --raw-input "data/raw/8cycle*.xlsx" \
  --dataset-name 8cycle
```

指定 `--raw-input` 后，脚本会先将外部 Excel 转换为 canonical CSV，再使用 checkpoint 中保存的特征列、标准化参数和模型权重，对这些序列整体评估。结果写入 `outputs/experiments/<experiment.name>/evaluation/<dataset-name>/`。

### 运行测试

```bash
python -m unittest discover -s tests -v
```

## 配置指南

### 基础配置结构

```yaml
# configs/base/default.yaml
seed: 42

experiment:
  name: default

data:
  format: canonical_csv        # 数据格式
  split_column: null            # null 表示按 sequence_id 随机划分
  window_size: 20              # 滑动窗口长度
  stride: 1                    # 滑动步长
  num_workers: 0               # DataLoader 读取进程数
  split_seed: 24               # 随机划分种子
  split:                       # 默认数据集划分比例
    train: 0.8
    val: 0.1
    test: 0.1

model:
  name: lstm                    # 编码器: lstm/gru/fcn/tcn/cnn
  hidden_size: 64
  num_layers: 2
  dropout: 0.0
  pooling:
    name: last                  # 池化: last/mean/max/attention
  head:
    hidden_size: null           # null 为直接 Linear
    dropout: 0.0

train:
  batch_size: 64
  epochs: 50
  learning_rate: 0.001
  optimizer:
    name: adam                  # adam/adamw/sgd
    weight_decay: 0.0
  loss:
    name: mse                   # mse/mae/l1/smooth_l1
  patience: 10                  # 早停耐心值
  min_delta: 0.0
  device: auto                  # auto/cuda/cpu

output:
  dir: outputs/experiments
```

### 实验配置（继承覆写）

```yaml
# configs/experiments/temp.yaml
extends:
  - ../base/default.yaml         # 继承基础配置

experiment:
  name: temp

data:
  dataset_name: own_cell_base_rates
  raw_path: data/raw/0.?C.xlsx   # 原始 Excel → 自动转换
  feature_columns:
    - voltage
    - current
    - temperature
    - power

model:
  name: lstm
  hidden_size: 64
  num_layers: 2
  dropout: 0.0
  pooling:
    name: last
  head:
    name: regression
    hidden_size: null
    dropout: 0.0

train:
  batch_size: 256
  epochs: 120
  learning_rate: 0.001
  optimizer:
    name: adam
    weight_decay: 0.0
  loss:
    name: mse
  patience: 10
```

各参数说明与可选值见 [`configs/README.md`](configs/README.md)。

## 扩展指南

### 添加新编码器

```python
from torch import nn
from src.models import register_encoder

class MyEncoder(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout):
        super().__init__()
        self.output_dim = hidden_size
        # ...

def build_my_encoder(config, input_dim):
    return MyEncoder(input_dim, config["hidden_size"], ...)

register_encoder("my_encoder", build_my_encoder)
# 然后在 YAML 中: model.name: my_encoder
```

### 添加新池化层

```python
from src.models import register_pooling

class MyPooling(nn.Module):
    def forward(self, encoded):
        return encoded[:, -1, :] * 2

register_pooling("my_pooling", lambda dim: MyPooling())
```

### 添加新损失函数

```python
from torch import nn
from src.training import register_loss

register_loss("huber", lambda config: nn.HuberLoss(delta=float(config.get("delta", 1.0))))
# YAML: train.loss.name: huber
```

### 添加新数据源转换器

在 `src/data/converters/` 下添加转换模块，将新数据源转换为 Canonical CSV；若希望 `train.py` 或 `eval.py` 自动处理该数据源，还需要在入口的数据准备分发逻辑中接入该转换器。当前内置自动转换流程仅处理 `cycler_workbook` Excel。

## 实验结果位置

每个实验的真实结果以本次运行生成的文件为准：

```text
outputs/experiments/<experiment.name>/
  resolved_config.yaml                实际生效的完整配置
  data_manifest.yaml                  本次使用的数据元信息
  history.json                        训练与验证损失历史
  summary.json                        最佳 epoch、验证损失与测试指标
  metrics.json                        MSE / MAE / RMSE / Max Error / R²
  predictions.csv                     每个测试窗口的真实值与预测值
  checkpoints/best.pt                 最佳模型检查点
  plots/                              训练与预测图
```

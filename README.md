# Battery SOC Estimation

基于深度学习的电池 **SOC（State of Charge，荷电状态）** 估计项目。从自有电池循环仪导出的 Excel 工作簿出发，完成数据预处理、模型训练、评估全流程。

## 项目结构

```
myproject/
├── configs/
│   ├── base/
│   │   └── default.yaml          # 基础配置（所有实验共享的默认值）
│   └── experiments/
│       ├── temp.yaml              # 实验配置示例（4 特征 LSTM）
│       └── test.yaml              # 实验配置示例（5 特征 LSTM）
├── data/
│   ├── raw/                       # 原始 Excel 工作簿 (*.xlsx)
│   └── processed/                 # 规范化 CSV（自动生成）
├── scripts/
│   ├── prepare_data.py            # 离线数据转换：Excel → 规范化 CSV
│   ├── train.py                   # 训练入口
│   └── eval.py                    # 评估入口
├── src/
│   ├── data/                      # 数据流水线
│   │   ├── converters/            # 自有设备 Excel 转换器
│   │   ├── dataset.py             # SOCDataset、DataBundle、数据加载器构建
│   │   ├── io.py                  # CSV 与 manifest 文件读写
│   │   ├── preprocess.py          # Z-score 标准化器
│   │   ├── schema.py              # 规范化数据契约与校验
│   │   └── window.py              # 滑动窗口构建
│   ├── models/                    # 模型组件
│   │   ├── encoders/              # 编码器：LSTM、GRU、FCN、CNN、TCN
│   │   ├── pooling/               # 池化：last、mean、max、attention
│   │   ├── head.py                # 回归头
│   │   ├── soc_model.py           # Encoder → Pooling → Head 组装
│   │   └── registry.py            # 模型组件注册表与构建器
│   ├── training/                  # 训练基础设施
│   │   ├── trainer.py             # Trainer（早停训练循环）+ predict
│   │   ├── checkpoint.py          # 检查点保存/加载
│   │   ├── losses.py              # 损失函数注册表（MSE/MAE/SmoothL1）
│   │   └── optimizers.py          # 优化器注册表（Adam/AdamW/SGD）
│   ├── evaluation/                # 评估
│   │   ├── metrics.py             # 回归指标（MSE/MAE/RMSE/MaxError/R²）
│   │   └── plots.py               # 可视化（训练曲线、预测对比图）
│   ├── experiment.py              # 实验管理（设备选择、评估保存）
│   └── utils/                     # 工具
│       ├── config.py              # YAML 配置加载与合并（支持 extends 继承）
│       ├── logger.py              # 统一日志记录器
│       ├── plugins.py             # 插件动态加载
│       └── seed.py                # 全局随机种子
├── tests/                         # 测试
├── outputs/                       # 训练产出（自动生成）
│   └── experiments/
│       └── <experiment_name>/
│           ├── checkpoints/
│           ├── plots/
│           ├── predictions.csv
│           ├── metrics.json
│           ├── summary.json
│           └── history.json
└── requirements.txt
```

## 架构设计

### 数据流水线

```
原始 Excel 工作簿               规范化 CSV                   滑动窗口                标准化特征
┌──────────────┐   离线转换     ┌──────────────┐   窗口化     ┌──────────────┐   fit on   ┌──────────────┐
│ .xlsx        │ ────────────→  │ sequence_id  │ ──────────→  │ (N, W, F)    │ ────────→  │ (N, W, F)    │
│ (record +    │               │ time         │              │ 窗口特征      │  train set │ 标准化后      │
│  auxTemp)    │               │ soc          │              │ + 目标值      │            │              │
└──────────────┘               │ voltage ...  │              └──────────────┘            └──────────────┘
                               └──────────────┘
```

关键设计决策：

- **标准化器仅对训练集拟合**，验证集和测试集使用训练集的均值和标准差做变换，避免数据泄露。
- **窗口以序列为边界**，不会跨序列滑动。窗口标签取窗口最后一个时间步的 SOC 值。
- **按序列划分数据集**（而非按行随机划分），确保同一充放电循环的数据不会同时出现在训练集和测试集中。
- **原始文件指纹**（SHA256 + 文件大小）写入 manifest，后续训练若原始文件未变则跳过转换。

#### SOC 计算逻辑

从循环仪 Excel 的 record 工作表中提取每个循环的充放电数据，通过安时积分法（Coulomb Counting）计算 SOC：

- 基于充/放电总容量和累积电量推导 SOC
- 1 Hz 去重采样（每秒保留一条记录）
- 温度从 auxTemp 工作表通过 record_id 映射
- 衍生特征：功率（voltage × current）、CC 容量（累积充电 - 累积放电）

### 模型架构

采用 **Encoder → Pooling → Head** 三段式可插拔架构：

```
输入 (batch, window_size, features)
        │
        ▼
┌───────────────────┐
│     Encoder        │  时序编码，提取每步特征
│  LSTM / GRU /      │  output: (batch, window_size, feature_dim)
│  FCN / CNN / TCN   │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│     Pooling         │  时间维聚合
│  last / mean /      │  output: (batch, feature_dim)
│  max / attention    │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│      Head           │  回归映射
│  RegressionHead     │  output: (batch, 1)
└───────────────────┘
```

#### 编码器说明

| 编码器 | 类型 | 特点 |
|--------|------|------|
| LSTM | 循环网络 | 长序列依赖建模，适合时序数据 |
| GRU | 循环网络 | 与 LSTM 类似，参数更少 |
| FCN | 全连接 | 逐时间步独立处理，渐进减半通道（hidden_size → hidden_size/2 → ...），每层含 BatchNorm1d |
| CNN | 一维卷积 | 沿时间轴提取局部模式，渐进减半通道，每层含 BatchNorm1d |
| TCN | 时序卷积 | 膨胀卷积扩大感受野 |

#### 池化策略说明

| 池化 | 行为 |
|------|------|
| last | 取最后一个时间步的输出 |
| mean | 对所有时间步取平均 |
| max | 对所有时间步取最大值 |
| attention | 学习每个时间步的权重，加权求和 |

#### 回归头

- `hidden_size` 为 null 时：单层 Linear(feature_dim, 1)
- `hidden_size` 非 null 时：Linear → ReLU → Dropout → Linear 两层结构

### 配置系统

配置采用两层继承机制：

```
configs/base/default.yaml       ← 所有实验的公共默认值
        ↑ extends
configs/experiments/*.yaml      ← 实验特定覆盖
```

实验配置文件中的 `extends` 字段指向一个或多个基础配置，最终配置由基础配置深度合并实验覆盖得到。配置中支持通过 `plugins` 字段声明要导入的外部模块（利用注册副作用扩展组件）。

### 训练流程

1. **数据准备**：检查原始 Excel 是否有对应的 canonical CSV 产物，若无则自动调用 `prepare_cycler_workbooks` 转换。
2. **数据加载**：加载所有 CSV → 按序列划分 train/val/test → 仅在训练集上拟合标准化器 → 构建滑动窗口 → 封装为 DataLoader。
3. **模型构建**：根据 `model` 配置节构建 Encoder + Pooling + Head。
4. **训练循环**：每个 epoch 后验证，保存最佳模型检查点。支持早停（`patience` + `min_delta`）。
5. **评估**：加载最佳检查点 → 测试集推理 → 计算 MSE/MAE/RMSE/MaxError/R² → 生成预测曲线图。

## 快速开始

### 环境准备

```bash
conda create -n SOC python=3.10
conda activate SOC
pip install -r requirements.txt
```

依赖项：PyTorch 2.12.0 (CUDA 12.3)、NumPy、Pandas、PyYAML、Matplotlib、openpyxl、tqdm。

### 1. 准备数据

将循环仪导出的 Excel 工作簿放入 `data/raw/` 目录。每个工作簿需包含：
- `record` 工作表：含数据序号、循环号、工步类型、时间、电流、电压等列
- `auxTemp` 工作表：含数据序号和温度列

数据转换会在训练时自动完成，也可手动运行：

```bash
python scripts/prepare_data.py \
    --input "data/raw/0.1C.xlsx" "data/raw/0.2C.xlsx" \
    --output data/processed/my_dataset
```

### 2. 创建实验配置

在 `configs/experiments/` 下创建 YAML 配置文件，继承基础配置并指定数据集和模型：

```yaml
extends:
  - ../base/default.yaml

experiment:
  name: my_lstm_experiment

data:
  dataset_name: my_dataset           # 对应 data/processed/<name>
  raw_path: data/raw/*.xlsx          # 原始 Excel 路径
  feature_columns:
    - voltage
    - current
    - temperature
    - power
    - cc_capacity

model:
  name: lstm                         # 编码器：lstm / gru / fcn / cnn / tcn
  hidden_size: 64
  num_layers: 2
  dropout: 0.1
  pooling:
    name: last                       # 池化：last / mean / max / attention
  head:
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
    name: mse                        # 损失函数：mse / mae / smooth_l1
  patience: 10
```

### 3. 训练

```bash
# 训练指定实验
python scripts/train.py --configs configs/experiments/my_experiment.yaml

# 训练多个实验（glob 模式）
python scripts/train.py --configs "configs/experiments/*.yaml"
```

训练产物输出到 `outputs/experiments/<experiment_name>/`，包含：
- `checkpoints/best.pt` — 最佳模型检查点
- `history.json` — 每轮训练/验证损失
- `metrics.json` — 测试集评估指标
- `predictions.csv` — 每条测试样本的预测值
- `plots/` — 训练曲线、预测对比图、各序列 SOC 时序图

### 4. 评估已有检查点

```bash
# 评估原始测试集
python scripts/eval.py --checkpoint outputs/experiments/<name>/checkpoints/best.pt

# 对外部新数据评估
python scripts/eval.py \
    --checkpoint outputs/experiments/<name>/checkpoints/best.pt \
    --raw-input "data/raw/new_battery.xlsx" \
    --dataset-name new_battery_test
```

## 配置参考

### data

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dataset_name` | str | — | 对应 `data/processed/<name>` 目录 |
| `raw_path` | str/list | — | 原始 Excel 路径或 glob |
| `feature_columns` | list | — | 模型输入特征列名 |
| `window_size` | int | 20 | 滑动窗口大小 |
| `stride` | int | 1 | 窗口滑动步长 |
| `split.train` | float | 0.8 | 训练集比例 |
| `split.val` | float | 0.1 | 验证集比例 |
| `split.test` | float | 0.1 | 测试集比例 |
| `split_column` | str | null | 使用数据中的列做划分（null 则随机划分） |
| `split_seed` | int | 24 | 数据集划分随机种子 |
| `num_workers` | int | 0 | DataLoader 工作进程数 |

### model

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | str | lstm | 编码器：lstm / gru / fcn / cnn / tcn |
| `hidden_size` | int | 64 | 隐藏层大小/通道数 |
| `num_layers` | int | 2 | 层数 |
| `dropout` | float | 0.0 | 编码器内 Dropout 比例 |
| `kernel_size` | int | 3 | 卷积核大小（仅 CNN、TCN） |
| `pooling.name` | str | last | 池化策略：last / mean / max / attention |
| `head.hidden_size` | int | null | 回归头隐层大小（null 为单层） |
| `head.dropout` | float | 0.0 | 回归头 Dropout 比例 |

### train

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `batch_size` | int | 64 | 批大小 |
| `epochs` | int | 50 | 最大训练轮数 |
| `learning_rate` | float | 0.001 | 学习率 |
| `optimizer.name` | str | adam | 优化器：adam / adamw / sgd |
| `optimizer.weight_decay` | float | 0.0 | 权重衰减 |
| `loss.name` | str | mse | 损失函数：mse / mae / smooth_l1 |
| `patience` | int | 10 | 早停耐心值 |
| `min_delta` | float | 0.0 | 判定改善的最小损失下降量 |
| `device` | str | auto | 计算设备：auto / cuda / cpu |
| `seed` | int | 42 | 全局随机种子 |

## 扩展

### 注册自定义组件

项目使用注册表模式，可以在外部模块中注册新的编码器、池化、回归头、损失函数或优化器，然后通过配置的 `plugins` 导入：

```python
# my_plugins.py
from src.models.registry import register_encoder
from src.training.losses import register_loss

register_encoder("my_encoder", my_encoder_builder)
register_loss("my_loss", my_loss_builder)
```

```yaml
# 配置中声明
plugins:
  - my_plugins
```

### 添加新特征

在 `feature_columns` 中添加列名即可，前提是规范化 CSV 中包含该列（若需从原始 Excel 提取，则修改 `cycler_workbook.py` 中的转换逻辑）。

# Battery SOC Estimation

这是一个面向电池荷电状态（SOC, State of Charge）估计的深度学习项目。主流程是把循环仪导出的 Excel 工作簿转换为统一的 canonical CSV，再按序列划分训练/验证/测试集，完成标准化、滑动窗口构造、模型训练和评估。

```text
data/raw/*.xlsx
  -> data/processed/<dataset_name>/sequences/*.csv
  -> optional downsampled processed dataset
  -> train/val/test split
  -> z-score normalization
  -> sliding windows
  -> model training / evaluation
```

## 项目结构

```text
myproject/
├── configs/
│   ├── base/default.yaml              # 公共默认配置
│   └── experiments/                   # 单个实验的覆盖配置
├── data/
│   ├── raw/                           # 原始 Excel 工作簿
│   └── processed/                     # 自动生成的 canonical CSV
├── scripts/
│   ├── prepare_data.py                # Excel -> canonical CSV
│   ├── downsample_data.py             # canonical CSV -> downsampled canonical CSV
│   ├── train.py                       # 训练入口
│   └── eval.py                        # 评估入口
├── src/
│   ├── data/                          # 数据转换、校验、标准化、滑窗、DataLoader
│   ├── models/                        # LSTM/GRU/CNN/TCN/双流模型等
│   ├── training/                      # Trainer、loss、optimizer、checkpoint
│   ├── evaluation/                    # 指标和绘图
│   ├── experiment.py                  # 实验输出与评估封装
│   └── utils/                         # 配置、日志、随机种子、插件加载
├── tests/                             # 单元测试
├── outputs/experiments/               # 训练和评估产物
└── requirements.txt
```

## 环境准备

推荐使用已经创建好的 Anaconda 环境 `SOC`。

```powershell
conda activate SOC
pip install -r requirements.txt
```

如果在非交互脚本或当前终端没有激活环境，也可以这样运行：

```powershell
conda run -n SOC python scripts/train.py --configs configs/experiments/lstm_25degC_0.5C_3loads.yaml
```

## 原始 Excel 要求

每个原始工作簿放在 `data/raw/` 下，当前转换器要求至少包含这些工作表和列：

| 工作表 | 原始列 | 转换后用途 |
|---|---|---|
| `record` | `绝对时间` | 用于时间对齐，并转换为相对秒 `time` |
| `record` | `电流(A)` | 转换为 `current`，充电为正、放电为负 |
| `record` | `电压(V)` | 转换为 `voltage` |
| `record` | `工步类型` | 判断充/放电阶段，用于电流符号和 SOC 计算 |
| `auxAdapter` | `绝对时间`, `PV1` | 按绝对时间对齐后得到 `force` |
| `auxTemp` | `绝对时间`, `T3` | 按绝对时间对齐后得到 `temperature` |

三张表按 `绝对时间` 对齐，不依赖 `record_id`。转换后会按 1 Hz 输出采样网格，并生成连续行号 `id=1,2,3,...`。

## canonical CSV 字段

转换后的 CSV 默认包含以下字段：

| 字段 | 来源或计算方式 |
|---|---|
| `id` | 采样后连续行号，从 1 开始 |
| `time` | `record["绝对时间"]` 转为相对秒，并按 1 Hz 采样输出 |
| `voltage` | `record["电压(V)"]` |
| `current` | `record["电流(A)"]`，充电正、放电负 |
| `power` | `voltage * current` |
| `cc_capacity` | 累计充电 Ah - 累计放电 Ah |
| `force` | `auxAdapter["PV1"]`，按绝对时间对齐 |
| `temperature` | `auxTemp["T3"]`，按绝对时间对齐 |
| `delta_f` | `force(t) - force(0)` |
| `delta_q` | `cc_capacity(t) - cc_capacity(0)` |
| `df_dt` | 相邻采样点的 `Δforce / Δtime`，分母接近 0 时为 0 |
| `df_dq` | 相邻采样点的 `Δforce / Δcc_capacity`，分母接近 0 时为 0 |
| `force_slope` | `delta_f / delta_q`，分母接近 0 时为 0 |
| `soc` | 保持当前库内逻辑：按完整充放电序列的累计充/放电容量归一化到 `[0, 1]` |
| `sequence_id` | 源 Excel 文件名 stem；训练划分和滑窗边界依赖该列 |

注意：`time` 是采样后的秒级网格标签。某一行可能来自原始时间 0.6 s 的记录，但输出 `time` 为 1.0 s，这是当前 1 Hz 采样策略的一部分。

## 手动转换数据

训练时如果配置了 `data.raw_path`，脚本会自动准备数据。也可以手动转换：

```powershell
conda run -n SOC python scripts/prepare_data.py `
  --input "data/raw/*.xlsx" `
  --output data/processed/own_cell_25degC_0.5C_3loads `
  --overwrite
```

生成内容：

```text
data/processed/<dataset_name>/
├── manifest.yaml
└── sequences/
    ├── <sequence_id_1>.csv
    ├── <sequence_id_2>.csv
    └── ...
```

`manifest.yaml` 会记录列名、序列数量、行数、采样周期、SOC 方法以及原始文件签名。训练脚本会用它判断 processed 数据是否仍然可复用。

## 从 canonical CSV 降采样

降采样从已经生成好的 `data/processed/<dataset_name>/sequences/*.csv` 开始，不重新解析 Excel。脚本会按每条序列的 `time` 时间网格抽样，保留抽样点的 `soc` 和基础传感器列，重新生成连续 `id`，并在降采样后的序列上重新计算 `delta_f`、`delta_q`、`df_dt`、`df_dq`、`force_slope`。

例如从 1s canonical 数据生成 5s 数据集：

```powershell
conda run -n SOC python scripts/downsample_data.py `
  --input data/processed/own_cell_25degC_0.5C_3loads `
  --output data/processed/own_cell_25degC_0.5C_3loads_5s `
  --interval-s 5 `
  --overwrite
```

常用采样间隔可以分别生成：

```powershell
conda run -n SOC python scripts/downsample_data.py `
  --input data/processed/own_cell_25degC_0.5C_3loads `
  --output data/processed/own_cell_25degC_0.5C_3loads_10s `
  --interval-s 10 `
  --overwrite

conda run -n SOC python scripts/downsample_data.py `
  --input data/processed/own_cell_25degC_0.5C_3loads `
  --output data/processed/own_cell_25degC_0.5C_3loads_30s `
  --interval-s 30 `
  --overwrite
```

生成后可以检查 `manifest.yaml`，确认 `sampling_period_s` 已变为目标采样间隔：

```powershell
Get-Content data/processed/own_cell_25degC_0.5C_3loads_5s/manifest.yaml
```

训练降采样数据时，把实验配置里的 `data.dataset_name` 改成对应的新数据集名，例如 `own_cell_25degC_0.5C_3loads_5s`。窗口大小 `window_size` 仍然表示时间步数量；如果希望保持相近的物理时间跨度，需要随着采样间隔调整它。

## 训练

当前项目的训练入口是：

```powershell
conda run -n SOC python scripts/train.py --configs configs/experiments/lstm_25degC_0.5C_3loads.yaml
```

也可以一次运行多个实验配置：

```powershell
conda run -n SOC python scripts/train.py --configs "configs/experiments/*.yaml"
```

训练产物默认输出到：

```text
outputs/experiments/<experiment_name>/
├── best.pt
├── predictions.csv
├── summary.json
└── plots/
    ├── soc_prediction.png
    ├── soc_error.png
    ├── pred_vs_true.png
    ├── soc_by_sequence.png
    └── gate_weights.png  # 仅门控双流模型生成
```

## 实验配置示例

实验配置通过 `extends` 继承 `configs/base/default.yaml`，只覆盖需要改变的部分。

```yaml
extends:
  - ../base/default.yaml

experiment:
  name: lstm_25degC_0.5C_3loads

data:
  dataset_name: own_cell_25degC_0.5C_3loads
  raw_path: data/raw/*.xlsx
  feature_columns:
    - voltage
    - current
    - temperature
    - cc_capacity
    - power
    - force
  window_size: 20
  stride: 1
  num_workers: 0

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
  epochs: 50
  learning_rate: 0.001
  optimizer:
    name: adam
    weight_decay: 0.0
  loss:
    name: mse
  patience: 10
```

`data.feature_columns` 决定模型实际输入哪些列。canonical CSV 中可以有更多列，例如 `delta_f`、`df_dt`、`delta_q`、`df_dq`、`force_slope`，只有写进 `feature_columns` 才会进入模型。

## 数据划分和预处理

- 数据按 `sequence_id` 划分 train/val/test，不按行随机划分。
- 默认至少需要 3 个不同的 `sequence_id`，否则无法同时得到 train、val、test。
- 标准化器只在训练集特征上拟合，验证集和测试集复用训练集均值/标准差，避免数据泄露。
- 滑动窗口不会跨越 `sequence_id` 边界。
- 窗口标签取窗口最后一个时间步的 `soc`。

## 模型

默认架构是：

```text
input window (batch, window_size, features)
  -> encoder
  -> pooling
  -> regression head
  -> SOC prediction
```

当前注册的常用组件：

| 配置项 | 可选值 |
|---|---|
| `model.name` | `lstm`, `gru`, `fcn`, `cnn`, `tcn`, `informer` |
| `model.pooling.name` | `last`, `mean`, `max`, `attention` |
| `model.head.name` | `regression` |
| `model.architecture.name` | 默认 `encoder_pooling_head`，也支持 `dual_stream` |
| `train.loss.name` | `mse`, `mae`, `smooth_l1` |
| `train.optimizer.name` | `adam`, `adamw`, `sgd` |

## 评估

评估已有 checkpoint：

```powershell
conda run -n SOC python scripts/eval.py `
  --checkpoint outputs/experiments/lstm_25degC_0.5C_3loads/best.pt
```

用已有 checkpoint 评估新的外部 Excel：

```powershell
conda run -n SOC python scripts/eval.py `
  --checkpoint outputs/experiments/lstm_25degC_0.5C_3loads/best.pt `
  --raw-input "data/raw/new_battery.xlsx" `
  --dataset-name new_battery_eval
```

外部评估会复用 checkpoint 中保存的特征列、标准化器和模型配置。

## 常用验证命令

```powershell
conda run -n SOC python -m unittest tests.test_cycler_workbook
conda run -n SOC python -m unittest tests.test_dataset
conda run -n SOC python -m unittest tests.test_train
conda run -n SOC python -m py_compile src/data/converters/cycler_workbook.py
```

## 常见注意点

- `data/raw/` 是原始数据目录，通常只放 Excel，不手工改动转换逻辑。
- `data/processed/` 和 `outputs/` 是生成产物，可以通过重新转换或重新训练再生成。
- 如果原始 Excel 文件数量、文件内容或 glob 范围变化，训练脚本会重新生成 processed 数据。
- 如果实验效果很差，先检查 train/val/test 是否只有很少序列。当前按序列划分时，3 条数据会变成训练、验证、测试各 1 条，泛化压力很大。
- 如果新增特征，通常需要同时更新 `src/data/converters/cycler_workbook.py`、相关测试，以及实验配置中的 `data.feature_columns`。

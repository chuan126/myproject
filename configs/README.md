# 配置说明

本目录只维护两类配置：

- `base/default.yaml`：稳定的工程默认值，通常不需要为每次实验重复修改。
- `experiments/*.yaml`：一次具体实验中希望被记录、比较或调节的参数。

运行一个实验：

```powershell
python scripts/train.py --configs configs/experiments/lstm_last_linear_adam_base_features.yaml
```

运行目录中的多个实验：

```powershell
python scripts/train.py --configs configs/experiments/*.yaml
```

## 推荐实验模板

```yaml
extends:
  - ../base/default.yaml

experiment:
  name: lstm_last_linear_adam_base_features
  run_name: auto

data:
  dataset_name: own_cell_base_rates
  raw_path: data/raw/0.?C.xlsx
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
  epochs: 50
  learning_rate: 0.001
  optimizer:
    name: adam
    weight_decay: 0.0
  loss:
    name: mse
  patience: 10
```

`dataset_name` 表示数据处理产物，`experiment.name` 表示模型实验身份。多个实验可以使用相同的 `dataset_name`，但应使用不同的 `experiment.name`。如果需要在同一组实验下保留多轮参数尝试，设置 `experiment.run_name: auto` 自动生成运行子目录。

## 顶层参数

| 参数 | 作用 | 取值 |
| --- | --- | --- |
| `extends` | 继承基础配置，实验文件只覆盖需要比较的字段 | 通常为 `../base/default.yaml` |
| `seed` | 控制随机初始化与可复现实验 | 非负整数，例如 `42` |
| `experiment.name` | 实验输出目录名称 | 唯一、可读的字符串 |
| `experiment.run_name` | 一次具体运行；为 `auto` 时使用本次训练命令共享的时间戳 | `null`、`auto` 或单级目录名 |
| `output.dir` | 所有实验结果的根目录 | 路径，例如 `outputs/experiments` |
| `output.overwrite` | 目标输出目录非空时是否允许覆盖 | `false` 或 `true`，默认 `false` |

## 数据参数 `data`

| 参数 | 作用 | 取值 |
| --- | --- | --- |
| `format` | 训练阶段读取的数据协议 | 当前仅支持 `canonical_csv` |
| `dataset_name` | 离线处理后的数据集名称 | 单级目录名称，例如 `own_cell_base_rates`；不要包含 `/` 或 `..` |
| `raw_path` | 待转换的原始 Excel 路径或 glob 模式 | 字符串或字符串列表，例如 `data/raw/0.?C.xlsx` |
| `path` | canonical CSV 路径；提供 `dataset_name` 时可省略 | 路径/glob；默认推导为 `data/processed/<dataset_name>/sequences/**/*.csv` |
| `manifest` | 数据集元数据文件；提供 `dataset_name` 时可省略 | 路径；默认推导为 `data/processed/<dataset_name>/manifest.yaml` |
| `feature_columns` | 输入模型的特征列 | 列名列表，必须存在于处理后的 CSV |
| `window_size` | 每个输入窗口包含的时间步数 | 正整数，例如 `20` |
| `stride` | 滑动窗口移动步长 | 正整数，例如 `1` |
| `num_workers` | `DataLoader` 读取进程数 | 非负整数；Windows 下建议从 `0` 开始 |
| `split_column` | 使用 CSV 中已有划分列 | `null` 或列名；列值必须为 `train`、`val`、`test` |
| `split_seed` | 未指定 `split_column` 时，按序列划分的随机种子 | 整数 |
| `split.train` | 训练序列占比 | `0` 到 `1` 的小数 |
| `split.val` | 验证序列占比 | `0` 到 `1` 的小数 |
| `split.test` | 测试序列占比 | `0` 到 `1` 的小数；三者合计应为 `1` |
| `extra_record_columns` | 从原始 `record` sheet 额外保留的输入候选列 | 列名列表，可选 |

当前 Excel 转换器要求原始工作簿包含 `record` 与 `auxTemp` sheet。常用特征列如下：

当配置含有 `raw_path` 时，训练入口会自动读取 `manifest` 校验 canonical CSV 的序列数、总行数、列结构，并核对原始 Excel 的文件集合与内容指纹。旧版完整产物首次被复用时会自动补全指纹记录，不重新转换 CSV。产物不存在、不完整或原始数据发生变化时，将自动重新离线处理，并移除该数据集目录中不再属于当前输入的旧序列 CSV；配置中不需要填写预处理开关。

| 特征列 | 含义 |
| --- | --- |
| `voltage` | 电压 |
| `current` | 电流 |
| `temperature` | 温度 |
| `power` | 功率 |
| `cc_capacity` | 库仑计量容量；若 SOC 标签由同一容量过程计算，使用它会使评估偏乐观，适合专门做消融比较 |

## 模型参数 `model`

| 参数 | 作用 | 取值 |
| --- | --- | --- |
| `architecture.name` | 完整模型组装方式 | 当前为 `encoder_pooling_head`；通常可省略 |
| `name` | 时序编码器类型 | `lstm`、`gru`、`fcn`、`tcn`、`cnn` |
| `hidden_size` | 编码器隐层或通道维度 | 正整数，例如 `64` |
| `num_layers` | 编码器堆叠层数 | 正整数，例如 `2` |
| `dropout` | 编码器内部 dropout 比例 | `0.0` 到小于 `1.0` 的小数 |
| `kernel_size` | 卷积核大小，仅 `tcn`、`cnn` 使用 | 正整数，例如 `3` |

### 池化参数 `model.pooling`

池化将时间序列编码结果汇聚为一个向量，再交给预测头输出 SOC。

| 参数 | 作用 | 取值 |
| --- | --- | --- |
| `name` | 时间维度汇聚方法 | `last`、`mean`、`max`、`attention` |

| 取值 | 含义 |
| --- | --- |
| `last` | 使用最后一个时间步的表示 |
| `mean` | 对窗口内所有时间步取平均 |
| `max` | 对窗口内所有时间步取最大响应 |
| `attention` | 学习不同时间步的重要程度 |

### 预测头参数 `model.head`

| 参数 | 作用 | 取值 |
| --- | --- | --- |
| `name` | SOC 回归预测头类型 | 当前支持 `regression` |
| `hidden_size` | 预测头中间层大小 | `null`/`0` 表示单层线性输出；正整数表示一层 MLP |
| `dropout` | 预测头隐藏层的 dropout 比例 | `0.0` 到小于 `1.0` 的小数；仅有隐藏层时生效 |

## 训练参数 `train`

| 参数 | 作用 | 取值 |
| --- | --- | --- |
| `batch_size` | 每个训练批次的窗口数 | 正整数，例如 `64`、`256` |
| `epochs` | 最大训练轮数 | 正整数 |
| `learning_rate` | 优化器学习率 | 正数，例如 `0.001` |
| `patience` | 验证损失连续未改善多少轮后早停 | 非负整数 |
| `min_delta` | 被视为改善所需的最小损失下降 | 非负数，例如 `0.0` |
| `device` | 训练设备 | `auto`、`cpu`、`cuda`、`cuda:0` 等 PyTorch device 字符串 |

### 优化器参数 `train.optimizer`

```yaml
optimizer:
  name: adam
  weight_decay: 0.0
```

| 参数 | 作用 | 取值 |
| --- | --- | --- |
| `name` | 优化器类型 | `adam`、`adamw`、`sgd` |
| `weight_decay` | 权重衰减强度 | 非负数，例如 `0.0`、`0.0001` |
| `momentum` | 动量，仅 `sgd` 使用 | `0.0` 到小于 `1.0` 的小数，例如 `0.9` |

为兼容旧配置，`optimizer: adam` 仍可运行，但新实验应使用对象形式，便于记录优化器参数。

### 损失函数参数 `train.loss`

```yaml
loss:
  name: mse
```

| 参数 | 作用 | 取值 |
| --- | --- | --- |
| `name` | 回归损失类型 | `mse`、`mae`、`l1`、`smooth_l1` |
| `beta` | `smooth_l1` 的二次误差区域阈值 | 正数，仅在 `name: smooth_l1` 时使用；默认 `1.0` |

`mae` 与 `l1` 等价。为兼容旧配置，`loss: mse` 仍可运行。

## 输出产物

训练默认只保留精简产物：

```text
outputs/experiments/<experiment.name>/
├── best.pt
├── summary.json
├── predictions.csv
└── plots/
```

如果配置了 `experiment.run_name`，产物会插入到 `experiment.name` 的第一段之后：

```text
outputs/experiments/<experiment.name.first_part>/<experiment.run_name>/<experiment.name.remaining_parts>/
```

例如 `experiment.name: paper/table1_baselines/m1_lstm_uitf`、`experiment.run_name: auto` 会写入 `outputs/experiments/paper/<YYYYMMDD_HHMMSS>/table1_baselines/m1_lstm_uitf/`。

默认 `output.overwrite: false`，目标目录已有文件时训练会停止，避免覆盖旧结果。

完整配置、标准化器和数据划分会随 checkpoint 保存在 `best.pt` 中。

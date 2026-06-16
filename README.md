# Battery SOC Estimation

本项目用于储能电池 SOC 估计实验。当前主线是把循环仪导出的 Excel 工作簿转换为统一的 canonical CSV，再按完整序列划分训练、验证和测试集，完成标准化、滑动窗口构造、模型训练、评估和论文表格汇总。

核心论文方向见 [paper_roadmap.md](paper_roadmap.md)：

```text
面向储能电池 SOC 估计的力学感知双流 LSTM-Informer 门控融合方法
```

项目默认流程：

```text
data/raw/*.xlsx
  -> data/processed/<dataset_name>/sequences/*.csv
  -> train/val/test split by sequence_id
  -> z-score normalization fitted on train only
  -> sliding windows
  -> model training and test evaluation
  -> outputs/experiments/<experiment_name>/
```

## 项目结构

```text
myproject/
├── configs/
│   ├── base/default.yaml              # 全局默认配置
│   └── experiments/
│       ├── paper/                     # 17 个正式论文实验配置
│       └── paper_tune/                # 对应论文实验的 Optuna 调参配置
├── data/
│   ├── raw/                           # 原始 Excel 工作簿
│   └── processed/                     # canonical CSV 与降采样数据集
├── scripts/
│   ├── prepare_data.py                # Excel -> canonical CSV
│   ├── downsample_data.py             # canonical CSV -> lower-frequency CSV
│   ├── train.py                       # 训练入口
│   ├── eval.py                        # checkpoint 评估入口
│   ├── summarize_paper_results.py     # 从论文实验输出生成 paper_tables.md
│   └── tune.py                        # Optuna 调参入口
├── src/
│   ├── data/                          # 数据校验、转换、降采样、标准化、滑窗、DataLoader
│   ├── models/                        # 单流模型、双流模型、编码器、池化、融合、回归头
│   ├── training/                      # Trainer、loss、optimizer、checkpoint
│   ├── evaluation/                    # 指标与图表
│   ├── experiment.py                  # 评估和输出产物封装
│   └── utils/                         # 配置、日志、插件、随机种子
├── tests/                             # 单元测试与轻量集成测试
├── outputs/                           # 训练、评估和调参产物
├── paper_roadmap.md                   # 论文路线图
└── requirements.txt
```

`data/processed/` 和 `outputs/` 是生成产物目录，通常不手工编辑。

## 环境准备

项目使用 Python、PyTorch、pandas、openpyxl、PyYAML、matplotlib、tqdm 和 Optuna。当前推荐使用已经创建好的 Anaconda 环境 `SOC`。

```powershell
conda activate SOC
pip install -r requirements.txt
```

如果没有激活环境，也可以把下面所有 `python ...` 命令改成：

```powershell
conda run -n SOC python ...
```

## 原始 Excel 要求

原始工作簿放在 `data/raw/` 下。每个工作簿代表一条完整充放电序列，文件名 stem 会写入 canonical CSV 的 `sequence_id`。

当前转换器至少要求三张表：

| 工作表 | 必要列 | 转换后用途 |
|---|---|---|
| `record` | `绝对时间`, `电流(A)`, `电压(V)`, `工步类型` | 主时间轴、电流、电压、充放电阶段 |
| `auxAdapter` | `绝对时间`, `PV1` | 力信号 `force` |
| `auxTemp` | `绝对时间`, `T3` | 温度信号 `temperature` |

转换时按 `绝对时间` 对齐，不依赖 `record_id`。输出为 1 Hz 秒级网格，同一秒只保留第一条记录，`id` 会重新生成为 `1,2,3,...`。

建议文件名能表达工况，例如：

```text
2000N_25degC_0.5C.xlsx
3000N_25degC_0.1C.xlsx
0N_10degC_0.5C.xlsx
```

## canonical CSV 字段

转换后的 CSV 默认写入：

| 字段 | 来源或计算方式 |
|---|---|
| `id` | 采样后连续行号，从 1 开始 |
| `time` | `record["绝对时间"]` 转为相对秒，并按 1 Hz 网格输出 |
| `voltage` | `record["电压(V)"]` |
| `current` | `record["电流(A)"]`，充电为正，放电为负 |
| `power` | `voltage * current` |
| `cc_capacity` | 累计充电 Ah - 累计放电 Ah |
| `force` | `auxAdapter["PV1"]`，按绝对时间对齐到主时间轴 |
| `temperature` | `auxTemp["T3"]`，按绝对时间对齐到主时间轴 |
| `delta_f` | `force(t) - force(0)` |
| `delta_q` | `cc_capacity(t) - cc_capacity(0)` |
| `df_dt` | 相邻采样点 `delta force / delta time` |
| `df_dq` | 相邻采样点 `delta force / delta cc_capacity` |
| `force_slope` | `delta_f / delta_q` |
| `soc` | 由完整充放电容量归一化到 `[0, 1]` |
| `sequence_id` | 源 Excel 文件名 stem |

SOC 标签在转换阶段构造：

```text
充电阶段：当前累计充电容量 / 本序列总充电容量
放电阶段：1 - 当前累计放电容量 / 本序列总放电容量
```

## 数据处理契约

训练入口读取的是 canonical CSV。运行时至少要求：

- 必须包含 `time`、`soc`、`sequence_id`。
- `data.feature_columns` 中列出的特征必须全部存在，且是有限数值。
- `soc` 必须在 `[0, 1]` 内。
- 每个 `sequence_id` 内的 `time` 必须单调递增。
- train/val/test 必须在 `sequence_id` 级别划分，滑动窗口不能跨序列。
- 标准化器只在训练集上拟合，验证集和测试集复用训练集统计量。
- 窗口标签取窗口最后一个时间步的 `soc`。

## 手动准备数据

正式训练配置通常会自动准备数据；如果需要手动转换：

```powershell
python scripts/prepare_data.py --input "data/raw/*.xlsx" --output data/processed/data --overwrite
```

输出结构：

```text
data/processed/data/
├── manifest.yaml
└── sequences/
    ├── <sequence_id_1>.csv
    ├── <sequence_id_2>.csv
    └── ...
```

`manifest.yaml` 会记录列名、序列数、总行数、采样周期、SOC 方法和原始文件签名。训练脚本会用它判断 processed 数据是否可复用；原始 Excel 文件集合或内容变化时会自动重建。

## 降采样数据

降采样从已经生成的 canonical CSV 开始，不重新解析 Excel。脚本会按每条序列的 `time` 网格抽样，重新生成 `id`，重新计算 `power` 和力学派生特征。

论文表 4 使用三套降采样数据：

```powershell
python scripts/downsample_data.py --input data/processed/data --output data/processed/data_5s --interval-s 5 --overwrite
python scripts/downsample_data.py --input data/processed/data --output data/processed/data_10s --interval-s 10 --overwrite
python scripts/downsample_data.py --input data/processed/data --output data/processed/data_30s --interval-s 30 --overwrite
```

窗口长度 `data.window_size` 表示时间步数量，不是固定物理时长。`20` 个时间步在 `1s/5s/10s/30s` 数据中分别约等于 `20s/100s/200s/600s` 历史窗口。

## 模型

默认单流结构：

```text
input window -> encoder -> pooling -> regression head -> SOC
```

内置组件：

| 配置项 | 可选值 |
|---|---|
| `model.name` | `lstm`, `gru`, `fcn`, `cnn`, `tcn`, `informer` |
| `model.pooling.name` | `last`, `mean`, `max`, `attention` |
| `model.head.name` | `regression` |
| `train.loss.name` | `mse`, `mae`, `l1`, `smooth_l1` |
| `train.optimizer.name` | `adam`, `adamw`, `sgd` |

双流模型通过以下配置打开：

```yaml
model:
  architecture:
    name: dual_stream
```

双流结构会按 `model.feature_columns` 的顺序把输入拆成两个分支：

```text
main_branch -> encoder -> pooling
mech_branch -> encoder -> pooling
                         -> fusion -> regression head -> SOC
```

融合方式：

| `model.fusion.name` | 含义 |
|---|---|
| `concat` | 直接拼接两个分支表示 |
| `gated` | 学习向量门控 `g * main + (1 - g) * mech`，要求两个分支输出维度一致 |

论文主线的 DSMI-LI 配置使用 LSTM 主分支、Informer 力学分支和 gated fusion。

## 正式论文实验

正式实验配置在 `configs/experiments/paper/`，共有 17 个：

```text
table1_baselines/            表 1：3 个基线模型实验
table2_feature_ablation/     表 2：5 个输入特征消融实验
table3_structure_ablation/   表 3：5 个结构消融实验
table4_downsampling/         表 4：4 个采样间隔实验
```

共享配置为 `configs/experiments/paper/_base_paper_9seq.yml`。当前论文划分规则是：

```text
test:      sequence_id 精确匹配 2000N_25degC_0.5C
train/val: 其它 sequence_id 按 train=0.67, val=0.33 随机划分
```

所有 paper 配置默认：

```yaml
experiment:
  run_name: auto
output:
  overwrite: false
```

因此同一次 `train.py` 命令会共享同一个时间戳，并写到：

```text
outputs/experiments/paper/<YYYYMMDD_HHMMSS>/<table_name>/<experiment_name>/
```

### 推荐运行顺序

先运行不依赖降采样的表 1 至表 3：

```powershell
python scripts/train.py --configs configs/experiments/paper/table1_baselines/*.yaml
python scripts/train.py --configs configs/experiments/paper/table2_feature_ablation/*.yaml
python scripts/train.py --configs configs/experiments/paper/table3_structure_ablation/*.yaml
```

再生成降采样数据：

```powershell
python scripts/downsample_data.py --input data/processed/data --output data/processed/data_5s --interval-s 5 --overwrite
python scripts/downsample_data.py --input data/processed/data --output data/processed/data_10s --interval-s 10 --overwrite
python scripts/downsample_data.py --input data/processed/data --output data/processed/data_30s --interval-s 30 --overwrite
```

最后运行表 4：

```powershell
python scripts/train.py --configs configs/experiments/paper/table4_downsampling/*.yaml
```

如果已经确认 `data_5s`、`data_10s`、`data_30s` 都存在，可以一次性运行全部正式实验：

```powershell
python scripts/train.py --configs configs/experiments/paper/**/*.yaml
```

## 输出产物

每个训练实验只保留精简产物：

```text
outputs/experiments/<experiment_name>/
├── best.pt
├── summary.json
├── predictions.csv
└── plots/
    ├── soc_prediction.png
    ├── soc_error.png
    ├── pred_vs_true.png
    ├── soc_by_sequence.png
    └── gate_weights.png        # 仅 gated dual-stream 模型生成
```

`summary.json` 包含：

```text
best_epoch
best_val_loss
mse
mae
rmse
max_error
r2
```

`predictions.csv` 包含测试窗口的 `sequence_id`、`time`、`actual_soc`、`predicted_soc` 和 `error`。

如果目标输出目录已经有文件，默认会停止以避免覆盖。要保留多轮结果，优先使用 `experiment.run_name: auto`；只有明确要复用目录时才设置：

```yaml
output:
  overwrite: true
```

## 汇总论文表格

一轮 paper 实验完成后，用时间戳目录生成四张结果表：

```powershell
python scripts/summarize_paper_results.py --run-dir outputs/experiments/paper/<YYYYMMDD_HHMMSS>
```

脚本会读取该目录下 17 个实验的 `summary.json`，只生成一个文件：

```text
outputs/experiments/paper/<YYYYMMDD_HHMMSS>/paper_tables.md
```

该文件填入表 1 至表 4 的 `MAE` 和 `MSE`，每张表内独立加粗最优值。

## 评估 checkpoint

评估已有 checkpoint：

```powershell
python scripts/eval.py --checkpoint outputs/experiments/paper/<YYYYMMDD_HHMMSS>/table1_baselines/m3_dsmi_li/best.pt
```

默认输出到 checkpoint 同级目录下的 `evaluation/`。

也可以用已有 checkpoint 评估新的外部 Excel：

```powershell
python scripts/eval.py `
  --checkpoint outputs/experiments/paper/<YYYYMMDD_HHMMSS>/table1_baselines/m3_dsmi_li/best.pt `
  --raw-input "data/raw/new_sequence.xlsx" `
  --dataset-name new_sequence_eval
```

外部评估会复用 checkpoint 中保存的特征列、标准化器和模型配置。

## Optuna 调参

调参配置在 `configs/experiments/paper_tune/`。它们继承正式论文配置，只增加 `tuning.search_space`。

运行一个调参实验：

```powershell
python scripts/tune.py `
  --config configs/experiments/paper_tune/table1_baselines/m3_dsmi_li.yaml `
  --study-name table1_m3_dsmi_li `
  --trials 10
```

调参输出：

```text
outputs/tuning/<study-name>/
├── best_config.yaml
├── best_params.json
├── trials.csv
└── trials/
    └── trial_000/
        ├── config.yaml
        ├── best.pt
        └── summary.json
```

调参结果不直接作为论文结果。选好参数后，用 `best_config.yaml` 再跑正式训练：

```powershell
python scripts/train.py --configs outputs/tuning/table1_m3_dsmi_li/best_config.yaml
```

## 配置要点

配置通过 `extends` 深度合并，实验 YAML 只需要覆盖相对默认值不同的部分。

常用字段：

| 字段 | 作用 |
|---|---|
| `experiment.name` | 稳定实验身份，也决定输出目录 |
| `experiment.run_name` | 一次具体运行；`auto` 表示自动时间戳 |
| `output.dir` | 输出根目录，默认 `outputs/experiments` |
| `output.overwrite` | 目标目录已有文件时是否允许覆盖 |
| `data.dataset_name` | `data/processed/<dataset_name>` |
| `data.raw_path` | 原始 Excel 路径或 glob；存在时训练入口会自动准备数据 |
| `data.feature_columns` | 实际输入模型的特征列 |
| `data.split_rules` | 按 `sequence_id` 规则划分数据集 |
| `data.window_size` | 输入窗口长度，单位是时间步 |
| `data.stride` | 滑动窗口步长 |
| `model.architecture.name` | 默认单流；`dual_stream` 打开双流 |
| `train.batch_size` | 训练 batch size |
| `train.epochs` | 最大训练轮数 |
| `train.patience` | 早停耐心值 |

更完整的配置说明见 [configs/README.md](configs/README.md)；论文实验说明见 [configs/experiments/paper/README.md](configs/experiments/paper/README.md)；调参说明见 [configs/experiments/paper_tune/README.md](configs/experiments/paper_tune/README.md)。

## 常用验证命令

```powershell
python -m py_compile scripts/train.py scripts/eval.py scripts/summarize_paper_results.py
python -m unittest tests.test_cycler_workbook
python -m unittest tests.test_downsample
python -m unittest tests.test_dataset
python -m unittest tests.test_dual_stream_model
python -m unittest tests.test_informer_encoder
python -m unittest tests.test_train
python -m unittest tests.test_summarize_paper_results
```

需要快速确认完整训练链路时：

```powershell
python -m unittest tests.test_smoke
```

## 常见问题

- 找不到 raw Excel：检查 `data.raw_path` 是否能匹配到文件。
- 找不到 canonical CSV：检查 `data.dataset_name` 对应的 `data/processed/<dataset_name>` 是否存在，或配置是否包含 `raw_path` 以便自动转换。
- 测试集为空：检查 `sequence_id` 是否精确包含 `2000N_25degC_0.5C`，以及 `split_rules.test` 是否与文件名 stem 一致。
- 表 4 找不到数据：先生成 `data_5s`、`data_10s`、`data_30s`。
- 输出目录已存在：保留旧结果时使用 `experiment.run_name: auto`；确认要覆盖时设置 `output.overwrite: true`。
- gated fusion 报维度不匹配：两个分支编码器输出维度必须一致。
- 结果看起来异常好或异常差：先检查 `data.feature_columns` 是否包含与标签强相关的容量类特征，以及 train/val/test 是否按 `sequence_id` 而不是按窗口随机划分。

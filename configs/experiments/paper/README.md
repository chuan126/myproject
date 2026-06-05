# 论文实验配置与运行说明

本目录按论文结果表组织实验配置，用于复现 `paper_roadmap.md` 中规划的全部实验。

默认前提：你已经进入项目使用的 conda 环境，例如 `SOC`，因此下面命令都直接使用 `python`。

## 目录结构

```text
configs/experiments/paper/
  _base_paper_9seq.yml              论文实验共享配置
  table1_baselines/                 表 1：基线模型对比
  table2_feature_ablation/          表 2：输入特征消融
  table3_structure_ablation/        表 3：结构消融
  table4_downsampling/              表 4：采样间隔实验
```

当前共有 17 个正式实验配置：

```text
表 1：3 个实验，M1-M3
表 2：5 个实验，A1-A5
表 3：5 个实验，B1-B5
表 4：4 个实验，C1-C4
```

## 数据准备规则

原始 Excel 放在：

```text
data/raw
```

论文共享配置会自动读取：

```text
data/raw/*.xlsx
```

首次运行训练，或原始 Excel 文件变化后，训练脚本会自动转换 canonical CSV：

```text
data/raw/*.xlsx -> data/processed/data
```

不需要手动运行 `scripts/prepare_data.py`。

## 数据划分规则

配置文件使用 `sequence_id` 规则划分数据集：

```text
test:      sequence_id 匹配 *2000N*
train/val: 其它 sequence_id 按比例随机划分
```

默认比例：

```text
train: 0.67
val:   0.33
```

该划分由 `data.split_seed` 控制，可复现。  
如果 canonical CSV 中已经存在 `split` 列，项目会优先使用 `split` 列；如果没有 `split` 列，则使用上述 `split_rules`。

## 推荐完整运行流程

建议按下面顺序运行全部论文实验。不要一开始直接运行 `configs/experiments/paper/**/*.yaml`，因为表 4 的 C2-C4 依赖降采样数据集，需要先生成。

### 1. 运行表 1：基线模型对比

```powershell
python scripts/train.py --configs configs/experiments/paper/table1_baselines/*.yaml
```

这一步会自动完成 1 Hz canonical 数据转换，生成：

```text
data/processed/data
```

输出目录：

```text
outputs/experiments/paper/table1_baselines/
```

### 2. 运行表 2：输入特征消融

```powershell
python scripts/train.py --configs configs/experiments/paper/table2_feature_ablation/*.yaml
```

输出目录：

```text
outputs/experiments/paper/table2_feature_ablation/
```

### 3. 运行表 3：结构消融

```powershell
python scripts/train.py --configs configs/experiments/paper/table3_structure_ablation/*.yaml
```

输出目录：

```text
outputs/experiments/paper/table3_structure_ablation/
```

### 4. 生成表 4 所需降采样数据

表 4 的 C1 使用 1 Hz 数据集 `data/processed/data`。  
C2-C4 分别使用 5s、10s、30s 降采样数据集，运行前必须先生成：

```powershell
python scripts/downsample_data.py --input data/processed/data --output data/processed/data_5s --interval-s 5 --overwrite
python scripts/downsample_data.py --input data/processed/data --output data/processed/data_10s --interval-s 10 --overwrite
python scripts/downsample_data.py --input data/processed/data --output data/processed/data_30s --interval-s 30 --overwrite
```

生成后应存在：

```text
data/processed/data_5s
data/processed/data_10s
data/processed/data_30s
```

### 5. 运行表 4：采样间隔实验

```powershell
python scripts/train.py --configs configs/experiments/paper/table4_downsampling/*.yaml
```

输出目录：

```text
outputs/experiments/paper/table4_downsampling/
```

## 一次性运行表 1-3

如果只想先跑不依赖降采样的实验，可以运行：

```powershell
python scripts/train.py --configs configs/experiments/paper/table1_baselines/*.yaml configs/experiments/paper/table2_feature_ablation/*.yaml configs/experiments/paper/table3_structure_ablation/*.yaml
```

## 一次性运行全部实验

只有在已经生成 `data_5s`、`data_10s`、`data_30s` 后，才建议运行：

```powershell
python scripts/train.py --configs configs/experiments/paper/**/*.yaml
```

否则表 4 的 C2-C4 会因为找不到降采样数据集而失败。

## 每个实验的输出内容

每个实验会输出到：

```text
outputs/experiments/paper/<table_name>/<experiment_name>/
```

常见文件包括：

```text
best.pt
summary.json
predictions.csv
plots/soc_prediction.png
plots/soc_error.png
plots/pred_vs_true.png
plots/soc_by_sequence.png
plots/gate_weights.png    仅 gated dual-stream 模型生成
```

论文正文表格通常只需要 `summary.json` 中的：

```text
mae
mse
```

## 运行前检查清单

运行全部实验前，建议确认：

```text
1. data/raw 中只放入本轮论文实验要使用的 Excel。
2. 文件名中需要能识别 2000N，因为测试集规则依赖 *2000N*。
3. 每个 Excel 包含 record、auxAdapter、auxTemp 三个工作表。
4. 表 4 之前已经生成 data_5s、data_10s、data_30s。
5. 当前终端已经进入 SOC 环境。
```

## 常用故障排查

如果训练时提示找不到降采样数据：

```text
先运行第 4 步的 downsample_data.py 命令。
```

如果训练时重新转换了不想要的 Excel：

```text
检查 data/raw 中是否残留了旧实验 Excel。
```

如果测试集为空：

```text
检查 raw 文件名或 canonical CSV 的 sequence_id 是否包含 2000N。
```

如果 train/val/test 划分不符合预期：

```text
优先检查 canonical CSV 是否已有 split 列；有 split 列时会覆盖 split_rules。
```

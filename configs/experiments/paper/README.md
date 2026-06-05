# 论文实验配置

本目录按论文结果表组织实验配置，方便单独运行某一张表，也方便一次性复现实验矩阵。

共享基础配置默认从原始 Excel 自动准备 1 Hz 论文数据集：

```text
data/raw/*.xlsx -> data/processed/data
```

因此通常只需要把原始 Excel 放入：

```text
data/raw
```

训练脚本会在首次运行或原始文件变化后自动转换 canonical CSV。转换后的数据集位置为：

```text
data/processed/data
```

如果 canonical CSV 已经包含 `split` 列，项目会优先使用该列，取值为 `train`、`val` 或 `test`。

如果没有 `split` 列，论文配置会按 `sequence_id` 自动划分：

```text
test:  文件名匹配 *2000N*
train/val: 其它序列按比例随机划分
```

默认比例为：

```text
train: 0.67
val:   0.33
```

随机划分由 `data.split_seed` 控制，可复现。这样可以固定 `2000N` 作为测试工况，同时允许后续继续增加数据。新增的 `2000N` 序列默认进入测试集；其它新增序列会参与 train/val 划分。

目录对应关系：

```text
table1_baselines/           表 1：基线模型对比
table2_feature_ablation/    表 2：输入特征消融
table3_structure_ablation/  表 3：结构消融
table4_downsampling/        表 4：采样间隔实验
```

运行单张表：

```powershell
python scripts/train.py --configs configs/experiments/paper/table1_baselines/*.yaml
```

运行全部论文实验：

```powershell
python scripts/train.py --configs configs/experiments/paper/**/*.yaml
```

运行表 4 的 C2-C4 前，先生成降采样数据集：

```powershell
python scripts/downsample_data.py --input data/processed/data --output data/processed/data_5s --interval-s 5 --overwrite
python scripts/downsample_data.py --input data/processed/data --output data/processed/data_10s --interval-s 10 --overwrite
python scripts/downsample_data.py --input data/processed/data --output data/processed/data_30s --interval-s 30 --overwrite
```

训练输出会按 `experiment.name` 自动写入：

```text
outputs/experiments/paper/<table_name>/<experiment_name>
```

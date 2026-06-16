# 论文实验 Optuna 调参配置说明

本目录用于给 `configs/experiments/paper` 中的 17 个论文实验配置做 Optuna 超参数搜索。  
正式训练配置仍保留在 `configs/experiments/paper`；本目录只额外增加 `tuning.search_space`。

默认前提：你已经进入项目使用的 conda 环境，例如 `SOC`，因此下面命令都直接使用 `python`。

## 目录结构

```text
configs/experiments/paper_tune/
  _base_single_stream_tune.yml      单流模型搜索空间
  _base_dual_stream_tune.yml        双流模型搜索空间
  table1_baselines/                 表 1：基线模型调参配置
  table2_feature_ablation/          表 2：输入特征消融调参配置
  table3_structure_ablation/        表 3：结构消融调参配置
  table4_downsampling/              表 4：采样间隔实验调参配置
```

每个调参配置都继承对应的正式论文实验配置，例如：

```yaml
extends:
  - ../../paper/table1_baselines/m3_dsmi_li.yaml
  - ../_base_dual_stream_tune.yml
```

这样可以保证数据划分、特征列、模型结构和论文正式实验保持一致，只把搜索空间放在 `paper_tune`。

## 当前搜索空间

单流模型使用：

```yaml
tuning:
  direction: minimize
  metric: best_val_loss
  search_space:
    learning_rate:
      type: float
      low: 1.0e-5
      high: 3.0e-3
      log: true
      targets:
        - train.learning_rate
    weight_decay:
      type: float
      low: 1.0e-6
      high: 1.0e-2
      log: true
      targets:
        - train.optimizer.weight_decay
    hidden_size:
      type: categorical
      choices: [32, 64, 96, 128]
      targets:
        - model.hidden_size
    num_layers:
      type: int
      low: 1
      high: 3
      targets:
        - model.num_layers
    dropout:
      type: float
      low: 0.0
      high: 0.4
      targets:
        - model.dropout
    head_dropout:
      type: float
      low: 0.0
      high: 0.4
      targets:
        - model.head.dropout
```

双流模型使用同样的参数名，但 `hidden_size`、`num_layers`、`dropout` 会同时写入两个分支：

```yaml
targets:
  - model.main_branch.encoder.hidden_size
  - model.mech_branch.encoder.hidden_size
```

`scripts/tune.py` 不再写死搜索空间，而是读取当前实验配置里的：

```text
tuning.search_space
```

## 如何修改搜索空间

只改 YAML，不需要改代码。

如果要修改学习率范围，编辑：

```text
configs/experiments/paper_tune/_base_single_stream_tune.yml
configs/experiments/paper_tune/_base_dual_stream_tune.yml
```

例如：

```yaml
learning_rate:
  type: float
  low: 5.0e-5
  high: 1.0e-3
  log: true
  targets:
    - train.learning_rate
```

如果某个实验需要单独的搜索空间，可以直接在该实验 YAML 里覆盖对应参数：

```yaml
tuning:
  search_space:
    hidden_size:
      type: categorical
      choices: [64, 128]
      targets:
        - model.hidden_size
```

注意：YAML 深度合并会保留未覆盖的参数；只写 `hidden_size` 不会删除其它搜索参数。

## 支持的参数类型

当前支持三类：

```text
float        -> trial.suggest_float(name, low, high, log=...)
int          -> trial.suggest_int(name, low, high)
categorical  -> trial.suggest_categorical(name, choices)
```

每个搜索参数必须写 `targets`，用于声明最优值写回到哪些配置路径。

## 运行单个实验调参

先用少量 trial 做烟雾测试：

```powershell
python scripts/tune.py --config configs/experiments/paper_tune/table1_baselines/m3_dsmi_li.yaml --study-name m3_dsmi_li --trials 2
```

确认能跑通后再增加 trial 次数：

```powershell
python scripts/tune.py --config configs/experiments/paper_tune/table1_baselines/m3_dsmi_li.yaml --study-name m3_dsmi_li --trials 10
```

参数说明：

```text
--config      单个调参实验 YAML。
--study-name  本次调参输出目录名。
--trials      Optuna trial 次数，默认 10。
--output-dir  调参输出根目录，默认 outputs/tuning。
```

## 运行全部论文调参配置

`scripts/tune.py` 一次只接收一个配置。需要批量调参时，按表逐个运行。

如果要串行运行本目录下全部 17 个调参配置，可以直接运行：

```powershell
$trials = 10
Get-ChildItem configs/experiments/paper_tune/table*/*.yaml |
  Sort-Object FullName |
  ForEach-Object {
    $studyName = "$($_.Directory.Name)_$($_.BaseName)"
    python scripts/tune.py --config $_.FullName --study-name $studyName --trials $trials
  }
```

这会依次运行 `table1_baselines`、`table2_feature_ablation`、`table3_structure_ablation` 和 `table4_downsampling` 下的所有 YAML。  
如果只想跑某一张表，使用下面的逐表命令。

表 1：

```powershell
python scripts/tune.py --config configs/experiments/paper_tune/table1_baselines/m1_lstm_uitf.yaml --study-name table1_m1_lstm_uitf --trials 10
python scripts/tune.py --config configs/experiments/paper_tune/table1_baselines/m2_informer_uitf.yaml --study-name table1_m2_informer_uitf --trials 10
python scripts/tune.py --config configs/experiments/paper_tune/table1_baselines/m3_dsmi_li.yaml --study-name table1_m3_dsmi_li --trials 10
```

表 2：

```powershell
python scripts/tune.py --config configs/experiments/paper_tune/table2_feature_ablation/a1_uit.yaml --study-name table2_a1_uit --trials 10
python scripts/tune.py --config configs/experiments/paper_tune/table2_feature_ablation/a2_uitf.yaml --study-name table2_a2_uitf --trials 10
python scripts/tune.py --config configs/experiments/paper_tune/table2_feature_ablation/a3_uit_dfdq.yaml --study-name table2_a3_uit_dfdq --trials 10
python scripts/tune.py --config configs/experiments/paper_tune/table2_feature_ablation/a4_uitf_dfdq.yaml --study-name table2_a4_uitf_dfdq --trials 10
python scripts/tune.py --config configs/experiments/paper_tune/table2_feature_ablation/a5_uitf_xmech.yaml --study-name table2_a5_uitf_xmech --trials 10
```

表 3：

```powershell
python scripts/tune.py --config configs/experiments/paper_tune/table3_structure_ablation/b1_lstm_main.yaml --study-name table3_b1_lstm_main --trials 10
python scripts/tune.py --config configs/experiments/paper_tune/table3_structure_ablation/b2_lstm_all.yaml --study-name table3_b2_lstm_all --trials 10
python scripts/tune.py --config configs/experiments/paper_tune/table3_structure_ablation/b3_informer_all.yaml --study-name table3_b3_informer_all --trials 10
python scripts/tune.py --config configs/experiments/paper_tune/table3_structure_ablation/b4_dual_stream_concat.yaml --study-name table3_b4_dual_stream_concat --trials 10
python scripts/tune.py --config configs/experiments/paper_tune/table3_structure_ablation/b5_dsmi_li_gated.yaml --study-name table3_b5_dsmi_li_gated --trials 10
```

表 4：

```powershell
python scripts/tune.py --config configs/experiments/paper_tune/table4_downsampling/c1_1s_dsmi_li.yaml --study-name table4_c1_1s_dsmi_li --trials 10
python scripts/tune.py --config configs/experiments/paper_tune/table4_downsampling/c2_5s_dsmi_li.yaml --study-name table4_c2_5s_dsmi_li --trials 10
python scripts/tune.py --config configs/experiments/paper_tune/table4_downsampling/c3_10s_dsmi_li.yaml --study-name table4_c3_10s_dsmi_li --trials 10
python scripts/tune.py --config configs/experiments/paper_tune/table4_downsampling/c4_30s_dsmi_li.yaml --study-name table4_c4_30s_dsmi_li --trials 10
```

表 4 的 C2-C4 运行前必须先生成：

```text
data/processed/data_5s
data/processed/data_10s
data/processed/data_30s
```

## 调参输出

默认输出到：

```text
outputs/tuning/<study-name>/
  best_config.yaml
  best_params.json
  trials.csv
  trials/
    trial_000/
      config.yaml
      best.pt
      summary.json
```

其中：

```text
best_params.json  Optuna 找到的最佳参数。
best_config.yaml  将最佳参数写回后的正式训练配置。
trials.csv        每个 trial 的验证损失和参数。
```

## 用最佳参数正式训练

调参结果只用于选参数，不直接写入论文表格。  
调参完成后，用 `best_config.yaml` 再运行正式训练：

```powershell
python scripts/train.py --configs outputs/tuning/table1_m3_dsmi_li/best_config.yaml
```

正式训练完成后，再从正式训练输出目录的 `summary.json` 读取：

```text
mae
mse
```

并回填论文表格。

## 不建议放进搜索空间的内容

这些内容属于数据契约、数据划分或论文消融变量，不建议混入自动调参：

```text
data.window_size
data.stride
data.split_rules
data.feature_columns
model.feature_columns
model.main_branch.feature_columns
model.mech_branch.feature_columns
```

如果要比较这些内容，应新建明确的实验配置，而不是用 Optuna 随机搜索。

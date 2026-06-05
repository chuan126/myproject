# AGENTS.md

## 角色与沟通

- 默认用中文解释计划、取舍、验证结果和风险。
- 代码、测试、配置键、命令和 commit message 按项目现有英文/技术命名风格书写。
- 先理解项目结构和现有数据契约，再修改代码；不要凭空假设目录、字段或配置。
- 任务简单明确时直接完成；任务涉及数据契约、训练流程、模型结构或多文件改动时，先给简短计划。

## 项目概览

这是一个电池 SOC 估计项目，主流程为：

```text
data/raw/*.xlsx -> canonical CSV -> train/val/test split -> z-score -> sliding windows -> model training/eval
```

关键目录：

- `scripts/`: 命令入口，包含 `prepare_data.py`、`train.py`、`eval.py`。
- `src/data/`: 数据转换、校验、标准化、滑窗和 DataLoader。
- `src/models/`: 编码器、池化、回归头、双流模型和 registry。
- `configs/base/`: 默认配置。
- `configs/experiments/`: 实验配置。
- `tests/`: 单元测试。
- `data/raw/`: 原始 Excel。
- `data/processed/`、`outputs/`: 生成产物，通常不要手工编辑。

## 工作原则

- 做最小、精准、可审查的改动。
- 不要顺手重构无关代码。
- 优先复用现有函数、配置模式、测试风格和注册表机制。
- 不要无故新增依赖；新增生产依赖前必须说明理由、替代方案和维护风险。
- 不要修改 `.env`、密钥、凭据、私钥或本地机器配置。
- 不要删除用户已有代码或数据，除非用户明确要求。
- 可读写代码文件；对 `data/raw/`、`data/processed/`、`outputs/` 的写入要特别谨慎，并在回复中说明。

## 数据处理约定

- canonical CSV 必须满足 `src/data/schema.py` 的契约：至少包含 `time`、`soc`、`sequence_id`，以及配置中的 `data.feature_columns`。
- `sequence_id` 是训练管线按序列分组、划分数据集和构建滑窗的关键列；不要随意删除。
- `time` 应为数值秒，且在每个 `sequence_id` 内单调递增。
- `soc` 必须在 `[0, 1]` 范围内。
- 特征列必须是有限数值，不能包含 NaN/inf。
- 标准化器只在训练集上拟合，不能用验证/测试数据泄露统计量。
- 滑动窗口不能跨 `sequence_id` 边界，窗口标签取最后一个时间步的 `soc`。
- Excel 转换逻辑集中在 `src/data/converters/cycler_workbook.py`；新增或调整特征时同步更新测试。
- 当前 Excel 转换应保留训练所需 `sequence_id`，输出连续 `id` 可作为采样后行号。

## 模型与配置约定

- 实验配置优先通过 `configs/experiments/*.yaml` 覆盖 `configs/base/default.yaml`。
- 改模型结构前先检查 `src/models/registry.py` 和现有注册机制。
- 双流模型相关配置需要保持 `data.feature_columns`、`model.feature_columns`、分支 `feature_columns` 的顺序和数量一致。
- 不要擅自改训练/测试划分策略；如果必须改，说明对结果可比性和数据泄露风险的影响。

## 测试与验证

修改后尽量运行最小相关验证。
运行环境在anaconda创建的SOC里

优先级：

1. 相关单元测试。
2. `python -m py_compile` 做语法检查。
3. 相关脚本的轻量烟测。
4. 更大范围 unittest 或训练脚本。

常用命令：

```bash
python -m unittest tests.test_cycler_workbook
python -m unittest tests.test_dataset
python -m unittest tests.test_train
python -m py_compile src/data/converters/cycler_workbook.py
python scripts/prepare_data.py --input "data/raw/*.xlsx" --output data/processed/<dataset_name>
python scripts/train.py --configs configs/experiments/<experiment>.yaml
```

如果本地 Python 环境缺 `pandas`、`PyYAML`、`openpyxl`、`torch` 等依赖，明确说明缺失依赖和未能运行的命令，不要伪造测试结果。

## Git 规则

- 不要自动 commit，除非用户明确要求。
- 不要自动 push。
- 提交前必须先看 `git diff`。
- commit message 使用简洁 Conventional Commits，例如：
  - `fix: update workbook conversion mapping`
  - `feat: add dual stream model`
  - `test: cover cycler workbook conversion`
  - `docs: document data pipeline`
- 工作区可能已有用户改动；不要回退或覆盖无关改动。

## 输出格式

完成编码任务后，按以下格式总结：

```text
### 改了什么

- ...

### 验证

- 已运行：...
- 结果：...

### 风险 / 后续

- ...
```

如果只是解释、讨论或头脑风暴，不必强行使用该格式。

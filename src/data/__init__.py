"""数据层包入口。

本包负责 SOC 估计任务的全部数据处理流水线，包括：
- 数据加载与校验（io, schema）
- 数据格式转换（converters 子包：Excel 工作簿 → 规范化 CSV）
- 预处理（preprocess：标准化）
- 滑动窗口构建（window）
- 降采样（downsample）
- 数据集和 DataLoader 构建（dataset）

导出的公共 API 包括：
- 核心列常量：CORE_COLUMNS, SEQUENCE_COLUMN, SOC_COLUMN, TIME_COLUMN
- 数据容器：DataBundle, SOCDataset, WindowedData
- 预处理：Standardizer
- 流水线入口：build_dataloaders, build_evaluation_dataloader
- 窗口化：build_windows
- 数据准备：downsample_canonical_dataset, downsample_sequence_frame,
             prepare_cycler_workbook, prepare_cycler_workbooks
- 校验：validate_canonical_frame
"""

from .dataset import DataBundle, SOCDataset, build_dataloaders, build_evaluation_dataloader
from .downsample import downsample_canonical_dataset, downsample_sequence_frame
from .converters import (
    prepare_cycler_workbook,
    prepare_cycler_workbooks,
    raw_file_signature,
)
from .preprocess import Standardizer
from .schema import CORE_COLUMNS, SEQUENCE_COLUMN, SOC_COLUMN, TIME_COLUMN, validate_canonical_frame
from .window import WindowedData, build_windows

__all__ = [
    "CORE_COLUMNS",
    "DataBundle",
    "SEQUENCE_COLUMN",
    "SOC_COLUMN",
    "SOCDataset",
    "Standardizer",
    "TIME_COLUMN",
    "WindowedData",
    "build_dataloaders",
    "build_evaluation_dataloader",
    "build_windows",
    "downsample_canonical_dataset",
    "downsample_sequence_frame",
    "prepare_cycler_workbook",
    "prepare_cycler_workbooks",
    "raw_file_signature",
    "validate_canonical_frame",
]

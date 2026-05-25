"""回归评估指标。

计算 MSE、MAE、RMSE、Max Error、R² 五个常用回归指标。
"""

import numpy as np


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """计算回归评估指标。

    Args:
        actual: 真实值数组
        predicted: 预测值数组

    Returns:
        包含 mse、mae、rmse、max_error、r2 的字典
    """
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    errors = predicted - actual
    mse = float(np.mean(errors**2))
    mae = float(np.mean(np.abs(errors)))
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = float(1.0 - np.sum(errors**2) / denominator) if denominator else float("nan")
    return {
        "mse": mse,
        "mae": mae,
        "rmse": float(np.sqrt(mse)),
        "max_error": float(np.max(np.abs(errors))),
        "r2": r2,
    }

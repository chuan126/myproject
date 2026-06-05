"""回归评估指标计算模块。

本模块提供电池 SOC（State of Charge）预测任务中最常用的回归评估指标。
所有指标均基于 NumPy 实现，输入为真实值与预测值的数组，输出为标准化字典，
便于后续序列化存储和报表生成。

在整个项目中的角色：
  被 experiment.py 中的 evaluate_and_save 函数调用，对测试集预测结果进行
  定量评估，评估结果写入 JSON 指标文件供后续分析对比。
"""

import numpy as np


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """计算一组回归评估指标。

    一次性计算 MSE、MAE、RMSE、Max Error 和 R² 五个常用指标，
    避免多次遍历数组，提高计算效率。

    参数:
        actual: 一维真实值数组（SOC 真值），形状为 (N,)。
                传入后会被转为 float64 以确保数值精度。
        predicted: 一维预测值数组（模型输出的 SOC 预测值），形状与 actual 相同。
                   同样会被转为 float64。

    返回:
        包含以下键的字典，所有值均为 Python float 类型：
        - mse: 均方误差（Mean Squared Error），越小越好
        - mae: 平均绝对误差（Mean Absolute Error），对离群值不如 MSE 敏感
        - rmse: 均方根误差（Root Mean Squared Error），量纲与 SOC 一致
        - max_error: 最大绝对误差，反映最差情况下的预测偏差
        - r2: 决定系数（R²），取值范围 (-∞, 1]，1 表示完美预测；
              当真实值方差为零时（分母为零）返回 NaN

    注意事项:
        - 当所有真实值相等时，R² 的分母为零，此时返回 NaN 而非抛出异常
        - 所有指标均基于 float64 精度计算，确保与 NumPy 内部计算一致
    """
    # 统一转换为 float64，避免输入类型不一致导致的精度问题
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)

    # 计算逐元素误差
    errors = predicted - actual

    # 均方误差：误差平方的均值
    mse = float(np.mean(errors**2))
    # 平均绝对误差：误差绝对值的均值
    mae = float(np.mean(np.abs(errors)))

    # R² 计算：1 - (残差平方和 / 总平方和)
    # 总平方和 = sum((y_i - y_mean)^2)，度量真实值自身的离散程度
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    # 当分母为零时（所有真实值相同），R² 无定义，返回 NaN
    r2 = float(1.0 - np.sum(errors**2) / denominator) if denominator else float("nan")

    return {
        "mse": mse,
        "mae": mae,
        "rmse": float(np.sqrt(mse)),
        "max_error": float(np.max(np.abs(errors))),
        "r2": r2,
    }

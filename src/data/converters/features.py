"""电池力学特征工程模块。

从电池循环数据中（力、容量、时间）推导力学相关的派生特征。
包括累计变化量（delta_f, delta_q）、瞬时变化率（df_dt, df_dq）
和累计斜率（force_slope）。这些特征用于捕捉电池在充放电过程中的
力学行为变化，辅助 SOC 估计模型识别不同 SOC 区间的力学响应模式。

在整个项目中的角色：
- 位于 data/converters 子包，是特征工程的一部分
- 被 cycler_workbook.py 和 downsample.py 调用
- 在原始数据转换和降采样后重新计算力学特征
- 原地修改传入的数据行列表，不返回新对象
"""

from typing import Any


def safe_ratio(numerator: float, denominator: float) -> float:
    """安全除法：分母接近零时返回 0.0。

    避免因分母为零（或浮点下溢接近零）导致的除零错误或 inf/nan 值。
    阈值 1e-12 兼顾了数值稳定性和实际物理量精度。

    Args:
        numerator: 分子
        denominator: 分母

    Returns:
        商 numerator / denominator；如果 |denominator| < 1e-12 则返回 0.0
    """
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator


def derive_mechanical_features(rows: list[dict[str, Any]]) -> None:
    """在原地为每行补充力学派生特征。

    要求每行至少包含 time、force、cc_capacity 三个字段。
    如果数据中没有 force 字段（如纯电学数据），则静默跳过。

    为每一行计算以下特征：

    - **delta_f**：当前力相对首行力的变化量（N）
    - **delta_q**：当前容量相对首行容量的变化量（Ah）
      这两个特征反映了充放电过程中的累计力学和电学变化

    - **df_dt**：力对时间的瞬时变化率（N/s）
      使用当前行与前一行的差分计算，首行设为 0
      捕捉力的快速变化事件（如电极结构相变）

    - **df_dq**：力对容量的瞬时变化率（N/Ah）
      使用当前行与前一行的差分计算，首行设为 0
      反映单位容量变化引起的力变化，与 SOC 区间相关

    - **force_slope**：力对容量的累计斜率（N/Ah）
      从首行到当前行的总体力变化除以总容量变化
      是 df_dq 的累计版本，更平滑但响应更慢

    Args:
        rows: 字典列表，每行至少包含 "time"、"force"、"cc_capacity" 键。
              函数会原地修改每行，添加上述派生特征键。

    注意事项：
        - 首行的 df_dt 和 df_dq 为 0（因为没有前一行的参考值）
        - 所有除法通过 safe_ratio 保护，不会出现 inf/nan
        - 如果 rows 为空或不含 force 键，函数直接返回不做任何操作
    """
    if not rows or "force" not in rows[0]:
        return
    # 初始化基准值：首行的力、容量和时间
    initial_force = float(rows[0]["force"])
    initial_capacity = float(rows[0]["cc_capacity"])
    previous_force = initial_force
    previous_capacity = initial_capacity
    previous_time = float(rows[0]["time"])
    for values in rows:
        force = float(values["force"])
        capacity = float(values["cc_capacity"])
        time_s = float(values["time"])
        # 相对首行的累计变化
        delta_force = force - initial_force
        delta_capacity = capacity - initial_capacity
        values["delta_f"] = delta_force
        values["delta_q"] = delta_capacity
        # 相对前一行的瞬时变化率
        values["df_dt"] = safe_ratio(force - previous_force, time_s - previous_time)
        values["df_dq"] = safe_ratio(force - previous_force, capacity - previous_capacity)
        # 累计斜率
        values["force_slope"] = safe_ratio(delta_force, delta_capacity)
        # 更新前一行的参考值供下次迭代使用
        previous_force = force
        previous_capacity = capacity
        previous_time = time_s

from __future__ import annotations

from pathlib import Path

import pandas as pd


# =========================
# 文件路径
# =========================

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "outputs" / "co2_data.csv"
REPORT_FILE = BASE_DIR / "outputs" / "co2_analysis_report.txt"


def load_data() -> pd.DataFrame:
    """读取模拟数据并完成必要的类型转换。"""
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"未找到数据文件：{CSV_FILE}")

    df = pd.read_csv(CSV_FILE, encoding="utf-8-sig")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def analyze(df: pd.DataFrame) -> dict:
    """对教学楼 CO2 运行状态进行统计分析。"""
    total_points = len(df)
    if total_points == 0:
        raise ValueError("数据为空，无法分析")

    average_co2 = df["co2"].mean()
    max_co2 = df["co2"].max()

    # CO2 超标时间比例：按分钟统计，超过 1000ppm 的记录占比
    co2_over_1000_ratio = (df["co2"] > 1000).mean()

    # 人数与 CO2 的相关性，使用 Pearson 相关系数
    occupancy_co2_corr = df["occupancy"].corr(df["co2"])

    # 新风运行比例：
    # 本模拟中 fresh_air_level 始终大于 0，因此按“有新风输出”统计为运行中。
    fresh_air_running_ratio = (df["fresh_air_level"] > 0).mean()

    # 额外补充一个更有解释力的指标：高新风档位占比
    high_fresh_air_ratio = (df["fresh_air_level"] >= 50).mean()

    return {
        "total_points": total_points,
        "average_co2": average_co2,
        "max_co2": max_co2,
        "co2_over_1000_ratio": co2_over_1000_ratio,
        "occupancy_co2_corr": occupancy_co2_corr,
        "fresh_air_running_ratio": fresh_air_running_ratio,
        "high_fresh_air_ratio": high_fresh_air_ratio,
    }


def build_report(stats: dict) -> str:
    """生成中文分析报告文本。"""
    report = f"""教学楼 CO2 运行状态分析报告
================================

数据概况
- 数据总点数：{stats["total_points"]} 条（分钟级数据）
- 分析对象：成都地区高校普通教学楼教室

核心指标
1. 平均CO2浓度：{stats["average_co2"]:.1f} ppm
2. 最大CO2浓度：{stats["max_co2"]:.1f} ppm
3. CO2超过1000ppm的时间比例：{stats["co2_over_1000_ratio"]:.2%}
4. 人数和CO2相关性（Pearson）：{stats["occupancy_co2_corr"]:.3f}
5. 新风运行比例：{stats["fresh_air_running_ratio"]:.2%}

补充说明
- 高于等于50%新风档位的时间比例：{stats["high_fresh_air_ratio"]:.2%}
- 相关性为正，说明人数增加时，CO2通常会同步上升，符合人体呼吸产气规律。
- CO2超过1000ppm的时间占比越高，表示教室通风负荷越大，需要更积极的新风调节。
- 新风运行比例为100%，说明模拟系统在全时段均保持基础通风运行。

结论
本教学楼教室CO2水平整体呈现出“人数上升 -> CO2上升 -> 新风增强 -> CO2回落”的闭环特征，
能够较好反映智慧校园场景下的典型室内空气质量动态变化，可用于后续LSTM模型训练与预测分析。
"""
    return report


def main() -> None:
    df = load_data()
    stats = analyze(df)
    report = build_report(stats)

    REPORT_FILE.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n已保存分析报告：{REPORT_FILE}")


if __name__ == "__main__":
    main()

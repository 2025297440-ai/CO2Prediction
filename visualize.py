from __future__ import annotations

import os
import random
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / "work" / "mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# =========================
# 文件路径
# =========================

CSV_FILE = BASE_DIR / "outputs" / "co2_data.csv"
OUTPUT_FILE = BASE_DIR / "outputs" / "co2_daily_curve.png"


def configure_matplotlib() -> None:
    """配置中文字体和负号显示，避免图表中文乱码。"""
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_data() -> pd.DataFrame:
    """读取 CO2 模拟数据，并解析时间列。"""
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"未找到数据文件：{CSV_FILE}")

    df = pd.read_csv(CSV_FILE, encoding="utf-8-sig")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    return df


def pick_one_day(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """随机选择一天数据用于可视化。"""
    dates = df["date"].drop_duplicates().tolist()
    if not dates:
        raise ValueError("CSV 中没有可用日期数据")

    chosen_date = random.choice(dates)
    day_df = df[df["date"] == chosen_date].copy()
    day_df.sort_values("timestamp", inplace=True)
    return day_df, str(chosen_date)


def plot_daily_curves(day_df: pd.DataFrame, chosen_date: str) -> None:
    """绘制同一天的 CO2、人数和新风量曲线。"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # 1) CO2 浓度曲线
    axes[0].plot(day_df["timestamp"], day_df["co2"], color="#d62728", linewidth=1.8)
    axes[0].set_title(f"教学楼教室 CO2 日变化曲线（{chosen_date}）", fontsize=16)
    axes[0].set_ylabel("CO2浓度 (ppm)", fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.35)

    # 2) occupancy 人数曲线
    axes[1].plot(day_df["timestamp"], day_df["occupancy"], color="#1f77b4", linewidth=1.8)
    axes[1].set_title("同日教室人数变化曲线", fontsize=16)
    axes[1].set_ylabel("人数 (人)", fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.35)

    # 3) fresh_air_level 新风量曲线
    axes[2].plot(day_df["timestamp"], day_df["fresh_air_level"], color="#2ca02c", linewidth=1.8)
    axes[2].set_title("同日新风量控制曲线", fontsize=16)
    axes[2].set_ylabel("新风量 (%)", fontsize=12)
    axes[2].set_xlabel("时间", fontsize=12)
    axes[2].grid(True, linestyle="--", alpha=0.35)

    for ax in axes:
        ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    df = load_data()
    day_df, chosen_date = pick_one_day(df)
    plot_daily_curves(day_df, chosen_date)
    print(f"已随机选择日期：{chosen_date}")
    print(f"已保存图片：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()

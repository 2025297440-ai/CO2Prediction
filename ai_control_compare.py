from __future__ import annotations

import os
import random
from pathlib import Path

import matplotlib

# 使用无界面绘图后端，保证脚本可以在命令行环境直接保存图片。
BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / "work" / "mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn


# =========================
# 文件路径
# =========================

CSV_FILE = BASE_DIR / "outputs" / "co2_data.csv"
MODEL_FILE = BASE_DIR / "outputs" / "best_lstm_model.pth"
FIGURE_FILE = BASE_DIR / "outputs" / "ai_control_compare.png"
REPORT_FILE = BASE_DIR / "outputs" / "ai_control_compare_report.txt"


# =========================
# 教室与控制参数
# =========================

OUTDOOR_CO2 = 420.0
CO2_GENERATION_PER_PERSON = 1.05
MAX_AIR_CHANGE_PER_HOUR = 6.0
INPUT_WINDOW = 60
PREDICT_HORIZON = 15
RANDOM_SEED = 20260709


class LSTMRegressor(nn.Module):
    """与 train.py 中保持一致的 LSTM 回归模型。"""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.regressor(last_hidden).squeeze(-1)


def configure_matplotlib() -> None:
    """配置中文字体，避免图表中文乱码。"""
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_data() -> pd.DataFrame:
    """读取模拟教学楼数据。"""
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"未找到数据文件：{CSV_FILE}")

    df = pd.read_csv(CSV_FILE, encoding="utf-8-sig")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_model(device: torch.device) -> tuple[LSTMRegressor, dict]:
    """加载训练好的 LSTM 模型和归一化参数。"""
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"未找到模型文件：{MODEL_FILE}")

    # 该模型由本项目 train.py 生成，checkpoint 中包含 numpy scaler。
    checkpoint = torch.load(MODEL_FILE, map_location=device, weights_only=False)
    model = LSTMRegressor(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["scaler"]


def control_rule(co2_value: float) -> int:
    """传统和 AI 控制都使用同一套新风档位规则。"""
    if co2_value < 800:
        return 20
    if co2_value <= 1000:
        return 50
    return 100


def update_co2(current_co2: float, occupancy: int, fresh_air_level: int) -> float:
    """使用简化室内 CO2 守恒模型更新下一分钟浓度。"""
    co2_generation = occupancy * CO2_GENERATION_PER_PERSON
    air_change_per_hour = MAX_AIR_CHANGE_PER_HOUR * (fresh_air_level / 100.0)
    ventilation_factor = air_change_per_hour / 60.0
    co2_decay = ventilation_factor * max(0.0, current_co2 - OUTDOOR_CO2)
    next_co2 = current_co2 + co2_generation - co2_decay
    return float(np.clip(next_co2, OUTDOOR_CO2, 3000.0))


def predict_future_co2(
    history_rows: list[list[float]],
    model: LSTMRegressor,
    scaler: dict,
    device: torch.device,
) -> float:
    """使用过去 60 分钟数据预测未来 15 分钟后的 CO2。"""
    feature_mean = np.asarray(scaler["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(scaler["feature_std"], dtype=np.float32)
    target_mean = float(scaler["target_mean"])
    target_std = float(scaler["target_std"])

    x = np.asarray(history_rows[-INPUT_WINDOW:], dtype=np.float32)
    x_norm = (x - feature_mean) / feature_std
    x_tensor = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_norm = model(x_tensor).cpu().item()

    return pred_norm * target_std + target_mean


def simulate_day(
    day_df: pd.DataFrame,
    strategy: str,
    model: LSTMRegressor | None = None,
    scaler: dict | None = None,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """按天模拟某一种控制策略下的 CO2 和新风量变化。"""
    rows = []
    history_rows: list[list[float]] = []
    current_co2 = float(day_df.iloc[0]["co2"])

    for _, row in day_df.iterrows():
        occupancy = int(row["occupancy"])
        temperature = float(row["temperature"])
        humidity = float(row["humidity"])

        if strategy == "traditional":
            control_co2 = current_co2
        elif strategy == "ai":
            if len(history_rows) >= INPUT_WINDOW and model is not None and scaler is not None and device is not None:
                control_co2 = predict_future_co2(history_rows, model, scaler, device)
            else:
                # 前 60 分钟历史数据不足时，先使用传统规则平稳过渡。
                control_co2 = current_co2
        else:
            raise ValueError(f"未知控制策略：{strategy}")

        fresh_air_level = control_rule(control_co2)

        rows.append(
            {
                "timestamp": row["timestamp"],
                "date": row["date"],
                "co2": current_co2,
                "occupancy": occupancy,
                "fresh_air_level": fresh_air_level,
            }
        )

        history_rows.append([current_co2, temperature, humidity, occupancy, fresh_air_level])
        current_co2 = update_co2(current_co2, occupancy, fresh_air_level)

    return pd.DataFrame(rows)


def simulate_all_days(df: pd.DataFrame, model: LSTMRegressor, scaler: dict, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    """分别模拟传统控制和 AI 预测控制。"""
    traditional_parts = []
    ai_parts = []

    for _, day_df in df.groupby("date", sort=True):
        day_df = day_df.sort_values("timestamp").reset_index(drop=True)
        traditional_parts.append(simulate_day(day_df, strategy="traditional"))
        ai_parts.append(simulate_day(day_df, strategy="ai", model=model, scaler=scaler, device=device))

    traditional_df = pd.concat(traditional_parts, ignore_index=True)
    ai_df = pd.concat(ai_parts, ignore_index=True)
    return traditional_df, ai_df


def calculate_metrics(control_df: pd.DataFrame) -> dict:
    """计算 CO2 超标比例、平均新风量和新风能耗估算。"""
    over_1000_ratio = (control_df["co2"] > 1000).mean()
    avg_fresh_air = control_df["fresh_air_level"].mean()

    # 风机能耗近似与风量三次方相关，这里输出相对能耗单位，便于策略间对比。
    energy = ((control_df["fresh_air_level"] / 100.0) ** 3).sum()

    return {
        "over_1000_ratio": float(over_1000_ratio),
        "avg_fresh_air": float(avg_fresh_air),
        "energy": float(energy),
    }


def build_report(traditional_metrics: dict, ai_metrics: dict) -> str:
    """生成中文对比报告。"""
    comfort_improvement = 0.0
    if traditional_metrics["over_1000_ratio"] > 0:
        comfort_improvement = (
            traditional_metrics["over_1000_ratio"] - ai_metrics["over_1000_ratio"]
        ) / traditional_metrics["over_1000_ratio"]

    energy_change = 0.0
    if traditional_metrics["energy"] > 0:
        energy_change = (ai_metrics["energy"] - traditional_metrics["energy"]) / traditional_metrics["energy"]

    report = f"""教学楼新风系统传统控制与AI预测控制对比报告
========================================

核心指标
1. 传统控制 CO2超过1000ppm时间比例：{traditional_metrics["over_1000_ratio"]:.2%}
2. AI预测控制 CO2超过1000ppm时间比例：{ai_metrics["over_1000_ratio"]:.2%}
3. 传统控制平均新风量：{traditional_metrics["avg_fresh_air"]:.2f}%
4. AI预测控制平均新风量：{ai_metrics["avg_fresh_air"]:.2f}%
5. 传统控制新风能耗估算：{traditional_metrics["energy"]:.2f} 相对单位
6. AI预测控制新风能耗估算：{ai_metrics["energy"]:.2f} 相对单位
7. 舒适度改善比例：{comfort_improvement:.2%}
8. AI控制能耗变化比例：{energy_change:.2%}

说明
- CO2超过1000ppm时间比例越低，表示室内空气质量越稳定。
- 平均新风量越高，通常意味着通风更积极，但也可能带来更高能耗。
- 新风能耗估算采用风量三次方近似，用于比较两种策略的相对能耗。
- 舒适度改善比例基于CO2超标时间减少幅度计算。
"""
    return report


def plot_compare(traditional_df: pd.DataFrame, ai_df: pd.DataFrame) -> str:
    """随机选择一天，绘制三张对比曲线。"""
    chosen_date = random.choice(traditional_df["date"].drop_duplicates().tolist())
    trad_day = traditional_df[traditional_df["date"] == chosen_date]
    ai_day = ai_df[ai_df["date"] == chosen_date]

    fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True)

    axes[0].plot(trad_day["timestamp"], trad_day["co2"], color="#1f77b4", linewidth=2)
    axes[0].axhline(1000, color="#d62728", linestyle="--", linewidth=1.5, label="1000ppm阈值")
    axes[0].set_title(f"传统控制CO2曲线（{chosen_date}）", fontsize=16)
    axes[0].set_ylabel("CO2浓度 (ppm)", fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()

    axes[1].plot(ai_day["timestamp"], ai_day["co2"], color="#2ca02c", linewidth=2)
    axes[1].axhline(1000, color="#d62728", linestyle="--", linewidth=1.5, label="1000ppm阈值")
    axes[1].set_title("AI预测控制CO2曲线", fontsize=16)
    axes[1].set_ylabel("CO2浓度 (ppm)", fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend()

    axes[2].step(
        trad_day["timestamp"],
        trad_day["fresh_air_level"],
        where="post",
        color="#1f77b4",
        linewidth=1.8,
        label="传统控制新风量",
    )
    axes[2].step(
        ai_day["timestamp"],
        ai_day["fresh_air_level"],
        where="post",
        color="#d62728",
        linewidth=1.8,
        linestyle="--",
        label="AI预测控制新风量",
    )
    axes[2].set_title("两种策略新风量变化对比", fontsize=16)
    axes[2].set_xlabel("时间", fontsize=12)
    axes[2].set_ylabel("新风量 (%)", fontsize=12)
    axes[2].set_ylim(0, 110)
    axes[2].grid(True, linestyle="--", alpha=0.35)
    axes[2].legend()

    for ax in axes:
        ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(FIGURE_FILE, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(chosen_date)


def main() -> None:
    configure_matplotlib()
    random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_data()
    model, scaler = load_model(device)
    traditional_df, ai_df = simulate_all_days(df, model, scaler, device)

    traditional_metrics = calculate_metrics(traditional_df)
    ai_metrics = calculate_metrics(ai_df)
    report = build_report(traditional_metrics, ai_metrics)
    REPORT_FILE.write_text(report, encoding="utf-8")

    chosen_date = plot_compare(traditional_df, ai_df)

    print(report)
    print(f"绘图日期：{chosen_date}")
    print(f"对比图已保存：{FIGURE_FILE}")
    print(f"分析报告已保存：{REPORT_FILE}")


if __name__ == "__main__":
    main()

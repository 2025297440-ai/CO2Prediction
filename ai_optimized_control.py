from __future__ import annotations

import os
import random
from collections import deque
from pathlib import Path

import matplotlib

# 使用无界面绘图后端，保证命令行环境可以直接保存图片。
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
FIGURE_FILE = BASE_DIR / "outputs" / "ai_optimized_compare.png"
REPORT_FILE = BASE_DIR / "outputs" / "ai_optimized_report.txt"


# =========================
# 仿真参数
# =========================

OUTDOOR_CO2 = 420.0
CO2_GENERATION_PER_PERSON = 1.05
MAX_AIR_CHANGE_PER_HOUR = 6.0
INPUT_WINDOW = 60
PREDICT_HORIZON = 15
CONTROL_DELAY_MINUTES = 2
RANDOM_SEED = 20260709


class LSTMRegressor(nn.Module):
    """与 train.py 中保持一致的 LSTM 回归模型结构。"""

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
    """读取教学楼 CO2 模拟数据。"""
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

    # 模型由本项目 train.py 生成，checkpoint 内包含 numpy scaler，因此显式关闭 weights_only。
    checkpoint = torch.load(MODEL_FILE, map_location=device, weights_only=False)
    model = LSTMRegressor(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["scaler"]


def update_co2(current_co2: float, occupancy: int, fresh_air_level: int) -> float:
    """使用简化室内空气质量守恒模型更新下一分钟 CO2。"""
    co2_generation = occupancy * CO2_GENERATION_PER_PERSON
    air_change_per_hour = MAX_AIR_CHANGE_PER_HOUR * (fresh_air_level / 100.0)
    ventilation_factor = air_change_per_hour / 60.0
    co2_decay = ventilation_factor * max(0.0, current_co2 - OUTDOOR_CO2)
    next_co2 = current_co2 + co2_generation - co2_decay
    return float(np.clip(next_co2, OUTDOOR_CO2, 3000.0))


def traditional_control(current_co2: float) -> int:
    """
    传统控制策略。

    实时 CO2 未超过 1000ppm 时维持 30% 基础新风；
    超过 1000ppm 后才提高到 100% 新风，体现被动响应特征。
    """
    if current_co2 > 1000:
        return 100
    return 30


def base_ai_control(predicted_co2: float) -> int:
    """按预测的未来 15 分钟 CO2 浓度给出基础新风量。"""
    if predicted_co2 < 800:
        return 10
    if predicted_co2 <= 1000:
        return 30
    if predicted_co2 <= 1200:
        return 60
    return 100


def next_level(level: int) -> int:
    """将新风档位提升一级。"""
    levels = [10, 30, 60, 100]
    index = levels.index(level)
    return levels[min(index + 1, len(levels) - 1)]


def previous_level(level: int) -> int:
    """将新风档位降低一级。"""
    levels = [10, 30, 60, 100]
    index = levels.index(level)
    return levels[max(index - 1, 0)]


def next_level_with_cap(level: int, cap: int = 60) -> int:
    """将新风档位提升一级，但不超过指定上限。"""
    return min(next_level(level), cap)


def optimized_ai_control(
    predicted_co2: float,
    current_co2: float,
    occupancy: int,
    temperature: float,
    humidity: float,
) -> int:
    """
    AI 预测优化控制策略。

    先根据未来 15 分钟 CO2 预测值确定基础新风量；
    再结合人数、温度、湿度进行小幅修正，兼顾空气质量和节能。
    """
    level = base_ai_control(predicted_co2)

    # 人数较多且预测值接近风险区间时，提前提高一档。
    if occupancy >= 50 and predicted_co2 >= 750:
        level = next_level_with_cap(level, cap=60)
    elif occupancy >= 35 and predicted_co2 >= 820:
        level = next_level_with_cap(level, cap=60)

    if occupancy >= 40 and predicted_co2 >= 700:
        level = max(level, 60)
    if occupancy >= 25 and current_co2 >= 820:
        level = max(level, 60)
    if occupancy >= 20 and predicted_co2 >= 900:
        level = max(level, 60)

    # 温湿度偏高且人员较多时，提高通风以改善体感舒适度。
    if occupancy >= 25 and (temperature >= 29.0 or humidity >= 78.0) and predicted_co2 >= 800:
        level = next_level_with_cap(level, cap=60)

    # 当前 CO2 已接近阈值时，增加一个安全修正，避免预测控制过度保守。
    if current_co2 >= 950 and occupancy >= 20:
        level = next_level_with_cap(level, cap=60)
    if current_co2 >= 1000:
        level = 100

    # 低人数、低风险状态下降低一档，减少无效通风能耗。
    if occupancy <= 8 and predicted_co2 < 850 and current_co2 < 850 and temperature < 29.0 and humidity < 78.0:
        level = previous_level(level)

    return level


def predict_future_co2(
    history_rows: list[list[float]],
    model: LSTMRegressor,
    scaler: dict,
    device: torch.device,
) -> float:
    """使用过去 60 分钟数据预测未来 15 分钟后的 CO2 浓度。"""
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
    """按天模拟一种控制策略下的 CO2、新风量和能耗变化。"""
    rows = []
    history_rows: list[list[float]] = []
    current_co2 = float(day_df.iloc[0]["co2"])

    if strategy == "traditional":
        initial_command = traditional_control(current_co2)
    else:
        initial_command = 30
    delay_queue: deque[int] = deque([initial_command] * CONTROL_DELAY_MINUTES, maxlen=CONTROL_DELAY_MINUTES)

    for _, row in day_df.iterrows():
        occupancy = int(row["occupancy"])
        temperature = float(row["temperature"])
        humidity = float(row["humidity"])

        actual_fresh_air = delay_queue[0]

        if strategy == "traditional":
            fresh_air_command = traditional_control(current_co2)
            predicted_co2 = np.nan
        elif strategy == "ai":
            if len(history_rows) >= INPUT_WINDOW and model is not None and scaler is not None and device is not None:
                predicted_co2 = predict_future_co2(history_rows, model, scaler, device)
                fresh_air_command = optimized_ai_control(predicted_co2, current_co2, occupancy, temperature, humidity)
            else:
                # 前 60 分钟历史不足时，使用稳健的 30% 新风平稳过渡。
                predicted_co2 = np.nan
                fresh_air_command = 30
        else:
            raise ValueError(f"未知控制策略：{strategy}")

        rows.append(
            {
                "timestamp": row["timestamp"],
                "date": row["date"],
                "co2": current_co2,
                "occupancy": occupancy,
                "temperature": temperature,
                "humidity": humidity,
                "fresh_air_level": actual_fresh_air,
                "fresh_air_command": fresh_air_command,
                "predicted_co2": predicted_co2,
            }
        )

        history_rows.append([current_co2, temperature, humidity, occupancy, actual_fresh_air])
        current_co2 = update_co2(current_co2, occupancy, actual_fresh_air)
        delay_queue.append(fresh_air_command)

    return pd.DataFrame(rows)


def simulate_all_days(df: pd.DataFrame, model: LSTMRegressor, scaler: dict, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    """分别模拟传统控制和 AI 预测优化控制。"""
    traditional_parts = []
    ai_parts = []

    for _, day_df in df.groupby("date", sort=True):
        day_df = day_df.sort_values("timestamp").reset_index(drop=True)
        traditional_parts.append(simulate_day(day_df, strategy="traditional"))
        ai_parts.append(simulate_day(day_df, strategy="ai", model=model, scaler=scaler, device=device))

    return pd.concat(traditional_parts, ignore_index=True), pd.concat(ai_parts, ignore_index=True)


def calculate_energy(control_df: pd.DataFrame) -> float:
    """估算新风运行相对能耗，风机能耗近似与风量三次方相关。"""
    return float(((control_df["fresh_air_level"] / 100.0) ** 3).sum())


def calculate_metrics(control_df: pd.DataFrame) -> dict:
    """计算控制策略评价指标。"""
    return {
        "over_1000_ratio": float((control_df["co2"] > 1000).mean()),
        "avg_co2": float(control_df["co2"].mean()),
        "avg_fresh_air": float(control_df["fresh_air_level"].mean()),
        "energy": calculate_energy(control_df),
    }


def build_report(traditional_metrics: dict, ai_metrics: dict) -> str:
    """生成中文分析报告。"""
    energy_saving_rate = 0.0
    if traditional_metrics["energy"] > 0:
        energy_saving_rate = (traditional_metrics["energy"] - ai_metrics["energy"]) / traditional_metrics["energy"]

    comfort_improvement = 0.0
    if traditional_metrics["over_1000_ratio"] > 0:
        comfort_improvement = (
            traditional_metrics["over_1000_ratio"] - ai_metrics["over_1000_ratio"]
        ) / traditional_metrics["over_1000_ratio"]

    report = f"""AI预测优化新风控制系统分析报告
================================

一、方法介绍
本实验基于教学楼 CO2 数字孪生数据和已训练的 LSTM 模型，构建 AI 预测优化新风控制系统。
系统使用过去 60 分钟的 CO2、温度、湿度、人数和新风量数据，预测未来 15 分钟后的 CO2 浓度，
并根据预测结果提前调整新风量，实现空气质量保障与节能运行之间的平衡。

二、控制逻辑
传统控制策略采用实时响应方式：当实时 CO2 不超过 1000ppm 时维持 30% 新风，超过 1000ppm 后提高至 100% 新风。
AI 预测优化控制策略采用预测响应方式：预测 CO2 <800ppm 时新风量为 10%，800-1000ppm 时为 30%，
1000-1200ppm 时为 60%，超过 1200ppm 时为 100%。同时，系统结合 occupancy、温度和湿度进行修正：
高人数或温湿度偏高时提前提高一档，低人数且低风险时降低一档，并模拟 2 分钟新风调整延迟。

三、实验结果
- 传统控制 CO2超过1000ppm时间比例：{traditional_metrics["over_1000_ratio"]:.2%}
- AI预测控制 CO2超过1000ppm时间比例：{ai_metrics["over_1000_ratio"]:.2%}
- 传统控制平均CO2浓度：{traditional_metrics["avg_co2"]:.2f} ppm
- AI预测控制平均CO2浓度：{ai_metrics["avg_co2"]:.2f} ppm
- 传统控制平均新风量：{traditional_metrics["avg_fresh_air"]:.2f}%
- AI预测控制平均新风量：{ai_metrics["avg_fresh_air"]:.2f}%
- 传统控制新风运行能耗估算：{traditional_metrics["energy"]:.2f} 相对单位
- AI预测控制新风运行能耗估算：{ai_metrics["energy"]:.2f} 相对单位
- 节能率：{energy_saving_rate:.2%}
- 舒适度改善比例：{comfort_improvement:.2%}

四、节能分析
传统控制在 CO2 已经超过阈值后才提高新风，容易出现滞后调节和短时间高风量运行。
AI 预测优化控制通过提前识别 CO2 上升趋势，以 30% 或 60% 中间档位进行预调节，
减少了 100% 高风量运行的时间，因此在降低 CO2 超标风险的同时，也降低了相对能耗。

五、对绿色建筑改造的意义
该策略体现了智慧校园绿色低碳改造中的“按需通风”和“预测控制”思想。
相比单纯依赖实时阈值的传统控制，AI 控制能够更早响应教室人员变化，减少空气质量波动，
并降低不必要的新风能耗，为教学楼空调与新风系统的低碳运行提供可落地的技术路径。
"""
    return report


def plot_compare(traditional_df: pd.DataFrame, ai_df: pd.DataFrame, traditional_metrics: dict, ai_metrics: dict) -> str:
    """随机选择一天绘制 CO2、新风量和能耗对比图。"""
    chosen_date = random.choice(traditional_df["date"].drop_duplicates().tolist())
    trad_day = traditional_df[traditional_df["date"] == chosen_date]
    ai_day = ai_df[ai_df["date"] == chosen_date]

    fig, axes = plt.subplots(3, 1, figsize=(14, 14))

    axes[0].plot(trad_day["timestamp"], trad_day["co2"], label="传统控制CO2", color="#1f77b4", linewidth=2)
    axes[0].plot(ai_day["timestamp"], ai_day["co2"], label="AI预测控制CO2", color="#2ca02c", linewidth=2)
    axes[0].axhline(1000, color="#d62728", linestyle="--", linewidth=1.3, label="1000ppm阈值")
    axes[0].set_title(f"传统控制与AI预测控制CO2曲线对比（{chosen_date}）", fontsize=16)
    axes[0].set_ylabel("CO2浓度 (ppm)", fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()

    axes[1].step(
        trad_day["timestamp"],
        trad_day["fresh_air_level"],
        where="post",
        label="传统新风量",
        color="#1f77b4",
        linewidth=1.8,
    )
    axes[1].step(
        ai_day["timestamp"],
        ai_day["fresh_air_level"],
        where="post",
        label="AI预测新风量",
        color="#d62728",
        linewidth=1.8,
        linestyle="--",
    )
    axes[1].set_title("传统新风量变化 vs AI预测新风量变化", fontsize=16)
    axes[1].set_ylabel("新风量 (%)", fontsize=12)
    axes[1].set_ylim(0, 110)
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend()

    labels = ["传统控制", "AI预测控制"]
    energies = [traditional_metrics["energy"], ai_metrics["energy"]]
    colors = ["#1f77b4", "#2ca02c"]
    axes[2].bar(labels, energies, color=colors, width=0.45)
    axes[2].set_title("两种策略新风能耗对比", fontsize=16)
    axes[2].set_ylabel("相对能耗单位", fontsize=12)
    axes[2].grid(True, axis="y", linestyle="--", alpha=0.35)
    for i, energy in enumerate(energies):
        axes[2].text(i, energy, f"{energy:.0f}", ha="center", va="bottom", fontsize=11)

    for ax in axes[:2]:
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

    chosen_date = plot_compare(traditional_df, ai_df, traditional_metrics, ai_metrics)

    print(report)
    print(f"绘图日期：{chosen_date}")
    print(f"对比图已保存：{FIGURE_FILE}")
    print(f"中文报告已保存：{REPORT_FILE}")


if __name__ == "__main__":
    main()

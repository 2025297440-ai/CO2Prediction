from __future__ import annotations

import os
import random
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
OUTPUT_FILE = BASE_DIR / "outputs" / "lstm_prediction_result.png"


class LSTMRegressor(nn.Module):
    """与 train.py 中一致的 LSTM 回归模型结构。"""

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
    """配置中文字体，避免中文标题和坐标轴乱码。"""
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_data() -> pd.DataFrame:
    """读取 CO2 数据，并按时间排序。"""
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

    # 模型由本项目 train.py 生成，checkpoint 内包含 numpy scaler，因此这里显式关闭 weights_only。
    checkpoint = torch.load(MODEL_FILE, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]
    scaler = checkpoint["scaler"]

    model = LSTMRegressor(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, scaler


def choose_predictable_day(df: pd.DataFrame, input_window: int, predict_horizon: int) -> tuple[pd.DataFrame, str]:
    """随机选择一天，并确保这一天的数据长度足够生成预测序列。"""
    dates = df["date"].drop_duplicates().tolist()
    valid_days = []

    for date in dates:
        day_df = df[df["date"] == date].copy()
        if len(day_df) >= input_window + predict_horizon:
            valid_days.append((date, day_df))

    if not valid_days:
        raise ValueError("没有找到足够长的一天数据，无法进行 60 分钟输入和 15 分钟预测。")

    chosen_date, day_df = random.choice(valid_days)
    day_df.sort_values("timestamp", inplace=True)
    day_df.reset_index(drop=True, inplace=True)
    return day_df, str(chosen_date)


def predict_one_day(day_df: pd.DataFrame, model: LSTMRegressor, scaler: dict, device: torch.device) -> pd.DataFrame:
    """使用过去 60 分钟数据，预测未来 15 分钟后的 CO2。"""
    feature_columns = scaler["feature_columns"]
    feature_mean = np.asarray(scaler["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(scaler["feature_std"], dtype=np.float32)
    target_mean = float(scaler["target_mean"])
    target_std = float(scaler["target_std"])
    input_window = int(scaler["input_window"])
    predict_horizon = int(scaler["predict_horizon"])

    features = day_df[feature_columns].to_numpy(dtype=np.float32)
    actual_co2 = day_df["co2"].to_numpy(dtype=np.float32)
    timestamps = day_df["timestamp"].to_numpy()

    pred_times = []
    pred_values = []
    actual_values = []

    max_start = len(day_df) - input_window - predict_horizon + 1

    with torch.no_grad():
        for start in range(max_start):
            end = start + input_window
            target_index = end + predict_horizon - 1

            x = features[start:end]
            x_norm = (x - feature_mean) / feature_std
            x_tensor = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0).to(device)

            pred_norm = model(x_tensor).cpu().item()
            pred_co2 = pred_norm * target_std + target_mean

            pred_times.append(timestamps[target_index])
            pred_values.append(pred_co2)
            actual_values.append(actual_co2[target_index])

    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(pred_times),
            "actual_co2": actual_values,
            "predicted_co2": pred_values,
        }
    )


def plot_prediction(result_df: pd.DataFrame, chosen_date: str) -> None:
    """绘制实际 CO2 曲线与 AI 预测 CO2 曲线。"""
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(
        result_df["timestamp"],
        result_df["actual_co2"],
        label="实际CO2浓度",
        color="#1f77b4",
        linewidth=2,
    )
    ax.plot(
        result_df["timestamp"],
        result_df["predicted_co2"],
        label="AI预测CO2浓度",
        color="#d62728",
        linewidth=2,
        linestyle="--",
    )

    ax.set_title(f"LSTM未来15分钟CO2预测结果（{chosen_date}）", fontsize=16)
    ax.set_xlabel("时间", fontsize=12)
    ax.set_ylabel("CO2浓度 (ppm)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    random.seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_data()
    model, scaler = load_model(device)
    day_df, chosen_date = choose_predictable_day(
        df,
        input_window=int(scaler["input_window"]),
        predict_horizon=int(scaler["predict_horizon"]),
    )
    result_df = predict_one_day(day_df, model, scaler, device)
    plot_prediction(result_df, chosen_date)

    mae = np.mean(np.abs(result_df["predicted_co2"] - result_df["actual_co2"]))
    print(f"已随机选择日期：{chosen_date}")
    print(f"当天预测点数：{len(result_df)}")
    print(f"当天预测MAE：{mae:.2f} ppm")
    print(f"预测结果图片已保存：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()

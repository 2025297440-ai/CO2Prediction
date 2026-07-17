from __future__ import annotations

import math
import os
import random
from pathlib import Path

import matplotlib

# 使用无界面的绘图后端，保证脚本在命令行环境也能直接保存图片。
BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / "work" / "mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# =========================
# 路径与训练参数
# =========================

CSV_FILE = BASE_DIR / "outputs" / "co2_data.csv"
MODEL_FILE = BASE_DIR / "outputs" / "best_lstm_model.pth"
LOSS_CURVE_FILE = BASE_DIR / "outputs" / "training_loss_curve.png"

FEATURE_COLUMNS = ["co2", "temperature", "humidity", "occupancy", "fresh_air_level"]
TARGET_COLUMN = "co2"

INPUT_WINDOW = 60
PREDICT_HORIZON = 15
BATCH_SIZE = 128
LEARNING_RATE = 0.001
RANDOM_SEED = 20260709


def set_seed(seed: int = RANDOM_SEED) -> None:
    """固定随机种子，便于复现实验结果。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class LSTMRegressor(nn.Module):
    """用于预测未来 CO2 浓度的 LSTM 回归模型。"""

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


def load_data() -> pd.DataFrame:
    """读取 CSV，并按时间排序。"""
    if not CSV_FILE.exists():
        raise FileNotFoundError(f"未找到数据文件：{CSV_FILE}")

    df = pd.read_csv(CSV_FILE, encoding="utf-8-sig")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def build_sequences(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    构造 LSTM 样本。

    输入：过去 60 分钟的多变量数据。
    输出：未来 15 分钟后的 CO2 浓度。
    """
    features = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    targets = df[TARGET_COLUMN].to_numpy(dtype=np.float32)

    x_list: list[np.ndarray] = []
    y_list: list[float] = []
    max_start = len(df) - INPUT_WINDOW - PREDICT_HORIZON + 1

    for start in range(max_start):
        end = start + INPUT_WINDOW
        target_index = end + PREDICT_HORIZON - 1
        x_list.append(features[start:end])
        y_list.append(targets[target_index])

    return np.asarray(x_list, dtype=np.float32), np.asarray(y_list, dtype=np.float32)


def split_data(x: np.ndarray, y: np.ndarray) -> tuple:
    """按时间顺序划分训练集、验证集、测试集：70%、20%、10%。"""
    n_samples = len(x)
    train_end = int(n_samples * 0.7)
    val_end = int(n_samples * 0.9)

    return (
        x[:train_end],
        y[:train_end],
        x[train_end:val_end],
        y[train_end:val_end],
        x[val_end:],
        y[val_end:],
    )


def normalize_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple:
    """使用训练集统计量进行归一化，避免验证集和测试集信息泄漏。"""
    feature_mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    feature_std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
    feature_std = np.where(feature_std == 0, 1.0, feature_std)

    target_mean = y_train.mean()
    target_std = y_train.std()
    target_std = target_std if target_std > 0 else 1.0

    x_train_norm = (x_train - feature_mean) / feature_std
    x_val_norm = (x_val - feature_mean) / feature_std
    x_test_norm = (x_test - feature_mean) / feature_std

    y_train_norm = (y_train - target_mean) / target_std
    y_val_norm = (y_val - target_mean) / target_std
    y_test_norm = (y_test - target_mean) / target_std

    scaler = {
        "feature_columns": FEATURE_COLUMNS,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "target_mean": float(target_mean),
        "target_std": float(target_std),
        "input_window": INPUT_WINDOW,
        "predict_horizon": PREDICT_HORIZON,
    }

    return x_train_norm, y_train_norm, x_val_norm, y_val_norm, x_test_norm, y_test_norm, scaler


def make_loader(x: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    """将 numpy 数据转换为 PyTorch DataLoader。"""
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def choose_epoch_count(train_size: int) -> int:
    """根据样本量自动设置 epoch，数据越多时训练轮数略少。"""
    if train_size >= 30000:
        return 25
    if train_size >= 10000:
        return 35
    return 50


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device) -> tuple[list, list]:
    """训练模型，并按验证集 loss 保存最优参数。"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    epochs = choose_epoch_count(len(train_loader.dataset))

    best_val_loss = math.inf
    train_losses: list[float] = []
    val_losses: list[float] = []
    patience = 6
    no_improve_count = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * batch_x.size(0)

        train_loss = train_loss_sum / len(train_loader.dataset)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:02d}/{epochs} | 训练Loss: {train_loss:.6f} | 验证Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": {
                        "input_size": len(FEATURE_COLUMNS),
                        "hidden_size": 64,
                        "num_layers": 2,
                        "dropout": 0.2,
                    },
                },
                MODEL_FILE,
            )
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"验证集连续 {patience} 轮未提升，提前停止训练。")
            break

    return train_losses, val_losses


def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    """计算数据集平均 loss。"""
    model.eval()
    loss_sum = 0.0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss_sum += loss.item() * batch_x.size(0)

    return loss_sum / len(loader.dataset)


def evaluate_metrics(model: nn.Module, test_loader: DataLoader, device: torch.device, target_mean: float, target_std: float) -> tuple[float, float]:
    """在测试集上计算反归一化后的 MAE 和 RMSE。"""
    model.eval()
    preds: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x).cpu().numpy()
            preds.append(pred)
            labels.append(batch_y.numpy())

    pred_norm = np.concatenate(preds)
    label_norm = np.concatenate(labels)

    pred_real = pred_norm * target_std + target_mean
    label_real = label_norm * target_std + target_mean

    mae = np.mean(np.abs(pred_real - label_real))
    rmse = np.sqrt(np.mean((pred_real - label_real) ** 2))
    return float(mae), float(rmse)


def plot_loss_curve(train_losses: list[float], val_losses: list[float]) -> None:
    """保存训练 loss 曲线。"""
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, label="训练Loss", linewidth=2)
    ax.plot(val_losses, label="验证Loss", linewidth=2)
    ax.set_title("LSTM训练Loss曲线", fontsize=16)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(LOSS_CURVE_FILE, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    df = load_data()
    x, y = build_sequences(df)
    x_train, y_train, x_val, y_val, x_test, y_test = split_data(x, y)

    x_train, y_train, x_val, y_val, x_test, y_test, scaler = normalize_data(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
    )

    train_loader = make_loader(x_train, y_train, shuffle=True)
    val_loader = make_loader(x_val, y_val, shuffle=False)
    test_loader = make_loader(x_test, y_test, shuffle=False)

    model = LSTMRegressor(input_size=len(FEATURE_COLUMNS)).to(device)
    train_losses, val_losses = train_model(model, train_loader, val_loader, device)
    plot_loss_curve(train_losses, val_losses)

    checkpoint = torch.load(MODEL_FILE, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 将归一化参数一并写回模型文件，便于后续预测阶段复用。
    checkpoint["scaler"] = scaler
    torch.save(checkpoint, MODEL_FILE)

    mae, rmse = evaluate_metrics(
        model,
        test_loader,
        device,
        target_mean=scaler["target_mean"],
        target_std=scaler["target_std"],
    )

    print("\n训练完成")
    print(f"训练样本数：{len(x_train)}，验证样本数：{len(x_val)}，测试样本数：{len(x_test)}")
    print(f"测试集 MAE：{mae:.2f} ppm")
    print(f"测试集 RMSE：{rmse:.2f} ppm")
    print(f"最佳模型已保存：{MODEL_FILE}")
    print(f"训练Loss曲线已保存：{LOSS_CURVE_FILE}")


if __name__ == "__main__":
    main()

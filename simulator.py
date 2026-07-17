from __future__ import annotations

import csv
import math
import random
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


# =========================
# 基础参数
# =========================

# 教室建筑参数
ROOM_AREA_M2 = 80
ROOM_HEIGHT_M = 3.6
ROOM_VOLUME_M3 = ROOM_AREA_M2 * ROOM_HEIGHT_M  # 288 m3

# 模拟周期
DAYS = 90
START_DATE = datetime(2026, 6, 1, 8, 0, 0)
DAY_START_MINUTE = 8 * 60
DAY_END_MINUTE = 18 * 60
MINUTES_PER_DAY = DAY_END_MINUTE - DAY_START_MINUTE

# 生成输出目录
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "co2_data.csv"

# 随机种子：保证每次运行结果稳定，便于训练和复现
RANDOM_SEED = 20260709


@dataclass
class ClassroomState:
    """保存教室的连续状态，便于按分钟递推模拟。"""

    true_co2: float = 650.0
    indoor_temp: float = 26.0
    indoor_humidity: float = 65.0
    delay_queue: deque = None

    def __post_init__(self) -> None:
        if self.delay_queue is None:
            # 新风控制延迟 2 分钟，因此用队列保存历史指令
            self.delay_queue = deque([10, 10], maxlen=2)


def clamp(value: float, low: float, high: float) -> float:
    """把数值限制在指定范围内。"""
    return max(low, min(high, value))


def get_occupancy_and_status(day_index: int, minute_of_day: int) -> tuple[int, str]:
    """
    根据日期和时间生成教学楼教室的人员与课程状态。

    规则：
    - 上课前 10 分钟学生开始进入
    - 上课期间人数保持稳定
    - 下课后快速离开
    - 午休期间人数接近 0
    """
    weekday = (START_DATE + timedelta(days=day_index)).weekday()

    # 周末大部分时间空闲，仅少量机动课程
    if weekday >= 5:
        if 10 * 60 <= minute_of_day < 12 * 60 and random.random() < 0.35:
            if minute_of_day < 11 * 60:
                return random.randint(18, 45), "weekend_class"
            return random.randint(0, 8), "weekend_exit"
        if 14 * 60 <= minute_of_day < 16 * 60 and random.random() < 0.25:
            if minute_of_day < 15 * 60:
                return random.randint(12, 35), "weekend_class"
            return random.randint(0, 6), "weekend_exit"
        return 0, "idle"

    # 工作日课程表：上午四个单元，下午四个单元
    # 课程开始时间整体向后平移 10 分钟，
    # 这样 8:00-8:09 会自然呈现“学生陆续进教室”的阶段。
    class_blocks = [
        (8 * 60 + 10, 9 * 60),
        (9 * 60 + 10, 10 * 60),
        (10 * 60 + 10, 11 * 60),
        (11 * 60 + 10, 12 * 60),
        (14 * 60 + 10, 15 * 60),
        (15 * 60 + 10, 16 * 60),
        (16 * 60 + 10, 17 * 60),
        (17 * 60 + 10, 18 * 60),
    ]

    # 午休期间基本无人
    if 12 * 60 <= minute_of_day < 14 * 60:
        return 0, "lunch_break"

    for start, end in class_blocks:
        enter_start = start - 10
        leave_end = end + 5

        if enter_start <= minute_of_day < start:
            # 课前 10 分钟：学生陆续进入
            progress = (minute_of_day - enter_start) / 10.0
            target = int(5 + progress * random.randint(30, 60))
            return clamp(target, 0, 60), "pre_class"

        if start <= minute_of_day < end:
            # 上课期间：人数相对稳定
            base = random.randint(25, 60)
            return base, "in_class"

        if end <= minute_of_day < leave_end:
            # 下课后：快速离开
            progress = (minute_of_day - end) / 5.0
            remaining = int((1.0 - progress) * random.randint(10, 35))
            return clamp(remaining, 0, 60), "post_class"

    return 0, "idle"


def get_outdoor_conditions(day_index: int, minute_of_day: int) -> tuple[float, float]:
    """
    模拟成都夏季室外气候。

    温度范围约 23-32℃，湿度范围约 55%-85%。
    使用日周期正弦波 + 小幅随机扰动，避免“过于死板”。
    """
    day_wave = math.sin(2 * math.pi * day_index / 30.0)
    minute_wave = math.sin(2 * math.pi * (minute_of_day - 14 * 60) / (24 * 60))

    outdoor_temp = 27.5 + 2.2 * day_wave + 2.8 * minute_wave + random.uniform(-0.7, 0.7)
    outdoor_humidity = 72 + 7 * math.sin(2 * math.pi * (minute_of_day - 6 * 60) / (24 * 60)) - 2.0 * day_wave
    outdoor_humidity += random.uniform(-3.0, 3.0)

    return clamp(outdoor_temp, 23.0, 32.0), clamp(outdoor_humidity, 55.0, 85.0)


def get_fresh_air_level(control_co2: float) -> int:
    """
    根据 CO2 浓度控制新风风量。

    说明：
    - 这里依据“控制器看到的 CO2”来确定目标风量
    - 实际执行会有 2 分钟延迟
    """
    if control_co2 < 700:
        return 10
    if control_co2 < 850:
        return 30
    if control_co2 < 950:
        return 50
    if control_co2 < 1050:
        return 70
    return 100


def simulate_one_minute(
    state: ClassroomState,
    day_index: int,
    minute_of_day: int,
) -> dict:
    """按分钟推进一次教室状态，并返回一行 CSV 记录。"""
    occupancy, lesson_status = get_occupancy_and_status(day_index, minute_of_day)
    outdoor_temp, outdoor_humidity = get_outdoor_conditions(day_index, minute_of_day)

    # 传感器读取会有 ±5ppm 误差，因此控制逻辑使用“测量值”
    measured_co2_for_control = state.true_co2 + random.uniform(-5, 5)
    target_fresh_air = get_fresh_air_level(measured_co2_for_control)

    # 新风系统有 2 分钟响应延迟：当前执行的是 2 分钟前下达的风量命令
    actual_fresh_air = state.delay_queue[0]
    state.delay_queue.append(target_fresh_air)

    # -------------------------
    # CO2 守恒模型
    # -------------------------
    # 人体呼吸产生 CO2：人数越多，上升越快
    # 这里将“每人每分钟产生的 ppm 增量”设为经验值，方便形成可训练的时序特征
    co2_generation = occupancy * 1.05

    # 新风稀释：风量越大，室内外浓度差被拉平得越快
    # 假设 100% 风量对应约 6 次/小时的等效换气
    air_change_per_hour = 6.0 * (actual_fresh_air / 100.0)
    ventilation_factor = air_change_per_hour / 60.0
    co2_decay = ventilation_factor * max(0.0, state.true_co2 - 420.0)

    # 真实 CO2 状态更新
    new_true_co2 = state.true_co2 + co2_generation - co2_decay
    state.true_co2 = clamp(new_true_co2, 420.0, 3000.0)

    # 输出层再叠加一次传感器随机误差：±5ppm
    output_co2 = clamp(state.true_co2 + random.uniform(-5, 5), 420.0, 3000.0)

    # -------------------------
    # 室内温湿度模型
    # -------------------------
    # 室内温度：受室外温度轻微影响，人员多时略有升高
    temp_drift = 0.03 * (outdoor_temp - state.indoor_temp) + 0.004 * occupancy - 0.002 * actual_fresh_air
    state.indoor_temp = clamp(state.indoor_temp + temp_drift + random.uniform(-0.08, 0.08), 23.0, 32.0)

    # 室内湿度：受室外湿度影响，人员和新风共同作用
    humidity_drift = 0.03 * (outdoor_humidity - state.indoor_humidity) + 0.015 * occupancy - 0.03 * actual_fresh_air
    state.indoor_humidity = clamp(state.indoor_humidity + humidity_drift + random.uniform(-0.4, 0.4), 45.0, 85.0)

    timestamp = START_DATE + timedelta(days=day_index, minutes=minute_of_day - DAY_START_MINUTE)

    return {
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "co2": round(output_co2, 1),
        "temperature": round(state.indoor_temp, 1),
        "humidity": round(state.indoor_humidity, 1),
        "occupancy": int(occupancy),
        "fresh_air_level": int(actual_fresh_air),
        "lesson_status": lesson_status,
        "outdoor_temperature": round(outdoor_temp, 1),
        "outdoor_humidity": round(outdoor_humidity, 1),
    }


def build_dataset() -> list[dict]:
    """生成 90 天的分钟级 CO2 时序数据。"""
    random.seed(RANDOM_SEED)
    state = ClassroomState()
    rows: list[dict] = []

    for day_index in range(DAYS):
        for minute_of_day in range(DAY_START_MINUTE, DAY_END_MINUTE):
            rows.append(simulate_one_minute(state, day_index, minute_of_day))

    return rows


def main() -> None:
    rows = build_dataset()
    headers = [
        "timestamp",
        "co2",
        "temperature",
        "humidity",
        "occupancy",
        "fresh_air_level",
        "lesson_status",
        "outdoor_temperature",
        "outdoor_humidity",
    ]

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"已生成数据文件：{OUTPUT_FILE}")
    print(f"总行数：{len(rows)}")
    print("时间范围：", rows[0]["timestamp"], "->", rows[-1]["timestamp"])


if __name__ == "__main__":
    main()

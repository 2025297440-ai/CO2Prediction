# CO2-AI-PROJECT：教学楼 CO2 数字孪生与 AI 预测优化新风控制系统

## 1. 项目背景

随着高校教学楼用能规模持续增长，室内空气质量与建筑节能之间的矛盾日益突出。传统新风系统通常采用固定风量或基于实时 CO2 阈值的被动控制方式，容易出现两类问题：

- 当教室人数快速增加时，新风系统响应滞后，CO2 浓度可能短时间超过舒适阈值。
- 当教室人数较少或课间无人时，系统仍可能维持较高新风量，造成不必要的风机能耗和冷热负荷浪费。

本项目面向智慧校园绿色低碳改造场景，构建了一个教学楼教室 CO2 数字孪生模拟与 AI 预测控制系统。系统通过模拟成都地区高校普通教室的 CO2 时序数据，训练 LSTM 神经网络预测未来 15 分钟 CO2 浓度，并进一步建立 AI 预测优化新风控制策略，实现空气质量保障与节能运行的综合优化。

## 2. 成都教学楼绿色低碳改造应用场景

项目模拟对象为成都地区高校普通教学楼教室，具有以下典型特征：

- 教室面积约 80 平方米，层高 3.6 米，体积约 288 立方米。
- 使用时间集中在每日 8:00-18:00。
- 人员密度随课程安排显著波动，上课前快速进入，下课后快速离开，午休期间接近无人。
- 成都夏季室外环境温湿度较高，室内空气品质控制需要兼顾 CO2、温度、湿度和新风能耗。

该场景适合用于研究教学楼新风系统的智能化改造，包括按需通风、预测控制、低碳运行和智慧校园运维决策。

## 3. 系统架构

本项目整体流程如下：

```text
教学楼参数与课程规律
        |
        v
CO2数字孪生模拟器 simulator.py
        |
        v
分钟级时序数据 outputs/co2_data.csv
        |
        +--------------------------+
        |                          |
        v                          v
数据分析 data_analysis.py     可视化 visualize.py
        |
        v
LSTM训练 train.py
        |
        v
预测模型 outputs/best_lstm_model.pth
        |
        +------------------------------+
        |                              |
        v                              v
预测可视化 predict_visualize.py   新风控制对比 ai_control_compare.py
                                       |
                                       v
                       AI预测优化控制 ai_optimized_control.py
```

系统包含四个核心层次：

- 数据层：生成 CO2、温度、湿度、人数、新风量等分钟级时序数据。
- 模型层：基于 LSTM 建立未来 15 分钟 CO2 预测模型。
- 控制层：构建传统控制、AI 预测控制和 AI 预测优化控制策略。
- 评估层：输出 CO2 超标比例、平均 CO2、新风能耗、节能率和舒适度改善比例。

## 4. 数据模拟方法

数据由 `simulator.py` 自动生成，输出文件为 `outputs/co2_data.csv`。模拟周期为连续 90 天，每天 8:00-18:00，每分钟一条记录，共 54,000 条数据。

生成字段包括：

- `timestamp`：时间戳
- `co2`：室内 CO2 浓度
- `temperature`：室内温度
- `humidity`：室内湿度
- `occupancy`：教室人数
- `fresh_air_level`：新风量档位
- `lesson_status`：课程状态
- `outdoor_temperature`：室外温度
- `outdoor_humidity`：室外湿度

CO2 模拟采用简化室内空气质量守恒思想：

```text
下一时刻CO2 = 当前CO2 + 人员呼吸产生量 - 新风稀释量
```

其中，人员越多，CO2 上升越快；新风量越大，CO2 稀释越快。模拟器还加入了成都夏季温湿度变化、大学课程作息规律、传感器误差和新风控制响应延迟，使数据更接近真实教学楼运行状态。

## 5. LSTM CO2预测模型原理

本项目使用 PyTorch 构建 LSTM 神经网络，用于预测未来 15 分钟后的 CO2 浓度。

模型输入为过去 60 分钟的多变量时序数据：

- `co2`
- `temperature`
- `humidity`
- `occupancy`
- `fresh_air_level`

模型输出为：

- 未来 15 分钟后的 `co2`

LSTM 适合处理 CO2 这类具有时间依赖性的序列数据。教室 CO2 浓度不仅受当前人数影响，还受过去一段时间的人员累积、新风稀释、温湿度环境和控制策略影响。LSTM 通过门控结构保留历史信息，能够学习“人数变化 - CO2 上升 - 新风响应 - CO2 回落”的动态规律。

训练过程包括：

- 自动读取 `outputs/co2_data.csv`
- 构造 60 分钟输入窗口和 15 分钟预测标签
- 使用训练集统计量进行归一化
- 按时间顺序划分训练集 70%、验证集 20%、测试集 10%
- 保存验证集效果最优模型到 `outputs/best_lstm_model.pth`
- 输出训练 Loss 曲线到 `outputs/training_loss_curve.png`

当前训练结果中，测试集误差约为：

```text
MAE：21.73 ppm
RMSE：34.68 ppm
```

## 6. AI预测新风控制策略

项目实现了三类控制策略。

传统控制策略：

```text
实时CO2未超过1000ppm：维持基础新风
实时CO2超过1000ppm：提高新风量
```

这种策略简单可靠，但存在滞后性，只有当 CO2 已经升高后才进行强通风。

AI预测控制策略：

```text
使用LSTM预测未来15分钟CO2
根据预测CO2提前调整新风量
```

AI预测优化控制策略由 `ai_optimized_control.py` 实现。其核心逻辑为：

```text
预测CO2 < 800ppm：新风量 = 10%
预测CO2 800-1000ppm：新风量 = 30%
预测CO2 1000-1200ppm：新风量 = 60%
预测CO2 > 1200ppm：新风量 = 100%
```

在此基础上，系统进一步引入：

- 人数 `occupancy` 修正：高人数时提前提高新风档位。
- 温湿度修正：温度或湿度偏高时适度提高新风，改善体感舒适度。
- 低风险节能修正：低人数、低 CO2 风险时降低新风档位。
- 2 分钟新风调整延迟：模拟真实新风系统从控制指令到实际执行的响应滞后。

该策略不是简单地“预测高就加大新风”，而是在空气质量风险和能耗之间进行折中，目标是减少 CO2 超标时间，同时降低高风量运行时间。

## 7. 文件说明

```text
CO2-AI-PROJECT/
├─ simulator.py                  # 生成教学楼CO2数字孪生模拟数据
├─ visualize.py                  # 随机选择一天绘制CO2、人数、新风量曲线
├─ data_analysis.py              # 分析CO2运行状态并生成中文报告
├─ train.py                      # 训练LSTM CO2预测模型
├─ predict_visualize.py          # 加载LSTM模型并绘制预测结果
├─ ai_control_compare.py         # 传统控制与AI预测控制对比实验
├─ ai_optimized_control.py       # AI预测优化新风控制系统
├─ README.md                     # 项目说明文档
└─ outputs/
   ├─ co2_data.csv               # 教学楼CO2模拟数据
   ├─ best_lstm_model.pth        # LSTM最优模型文件
   ├─ co2_daily_curve.png        # 单日CO2、人数、新风量曲线
   ├─ training_loss_curve.png    # LSTM训练Loss曲线
   ├─ lstm_prediction_result.png # LSTM预测结果图
   ├─ ai_control_compare.png     # 传统控制与AI预测控制对比图
   ├─ ai_optimized_compare.png   # AI优化控制对比图
   ├─ co2_analysis_report.txt    # CO2运行状态分析报告
   ├─ ai_control_compare_report.txt
   └─ ai_optimized_report.txt    # AI优化控制中文报告
```

## 8. 运行步骤

建议在项目根目录 `CO2-AI-PROJECT` 下依次运行。

安装依赖：

```bash
pip install pandas numpy matplotlib torch
```

生成模拟数据：

```bash
python simulator.py
```

绘制基础数据曲线：

```bash
python visualize.py
```

生成 CO2 运行状态分析报告：

```bash
python data_analysis.py
```

训练 LSTM 预测模型：

```bash
python train.py
```

可视化 LSTM 预测效果：

```bash
python predict_visualize.py
```

对比传统控制与 AI 预测控制：

```bash
python ai_control_compare.py
```

运行 AI 预测优化新风控制系统：

```bash
python ai_optimized_control.py
```

## 9. 实验结果总结

LSTM 预测模型能够较好学习教学楼教室 CO2 的动态变化规律，在测试集上取得了较低预测误差：

```text
MAE：21.73 ppm
RMSE：34.68 ppm
```

AI 预测优化新风控制系统的最新实验结果如下：

```text
传统控制 CO2超过1000ppm时间比例：31.90%
AI预测控制 CO2超过1000ppm时间比例：24.66%

传统控制平均CO2浓度：793.41 ppm
AI预测控制平均CO2浓度：866.60 ppm

传统控制平均新风量：52.26%
AI预测控制平均新风量：44.39%

传统控制新风运行能耗估算：18169.28 相对单位
AI预测控制新风运行能耗估算：13801.79 相对单位

节能率：24.04%
舒适度改善比例：22.70%
```

结果表明，AI 预测优化控制相较传统实时阈值控制能够提前识别 CO2 上升趋势，减少 CO2 超过 1000ppm 的时间比例，并降低平均新风量和相对能耗。该结果说明预测控制在教学楼绿色低碳改造中具有应用潜力，可为高校智慧校园的新风系统优化、低碳运维和室内环境质量提升提供技术支撑。

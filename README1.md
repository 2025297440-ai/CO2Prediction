# 基于LSTM的教学楼CO2浓度预测模型

## 项目简介

本项目利用LSTM神经网络建立教学楼CO2浓度预测模型。

通过历史CO2浓度、温度、湿度、人员数量、新风量数据，
预测未来15分钟CO2浓度变化趋势。


## 项目流程

1. simulator.py
生成模拟教学楼CO2数据

2. data_analysis.py
数据统计分析

3. train.py
训练LSTM预测模型

4. predict_visualize.py
进行CO2预测并绘制结果

5. ai_optimized_control.py
基于预测结果模拟新风优化控制


## 环境

Python 3.13

主要库：

- PyTorch
- numpy
- pandas
- matplotlib


## 运行方法

生成数据：

python simulator.py


训练模型：

python train.py


预测：

python predict_visualize.py


优化控制：

python ai_optimized_control.py

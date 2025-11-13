import pandas as pd
from Data_Handling import DataHandler, get_data_handler
from Backtest_Engine import BacktestEngine
from Strategy_Core import WeightBasedStrategy

# 1. 初始化数据处理器（全局单例，预加载所有数据）
file_path = r"D:\read\task\机器学习数据.pkl"
data_handler = DataHandler(file_path)
get_data_handler(file_path)  # 初始化全局实例

# 2. 初始化回测引擎
backtest_engine = BacktestEngine(
    data_handler=data_handler,
    strategy_class=WeightBasedStrategy,
    initial_cash=100000000
)

# 3. 运行回测
backtest_engine.run(
    start_date=pd.to_datetime('2021-09-02'),
    end_date=pd.to_datetime('2022-9-15')
)
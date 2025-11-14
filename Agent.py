# 博弈高手
from tqdm import tqdm,trange
from Data_Handling import DataHandler
from Performance_Analysis import PerformanceAnalysis  # 导入绩效分析类
import pandas as pd
import numpy as np

# 离散简化智能体环境
class DiscreteIndexEnvironment:
    def __init__(self, file_path):
        """
        初始化离散化智能体环境

        参数:
        file_path: CSV文件路径
        """
        # 读取CSV文件
        self.df = pd.read_csv(file_path)
        self.df['trade_date'] = pd.to_datetime(self.df['trade_date'], format='%Y%m%d')

        # 获取基准数据（2016-2018）
        start_date = pd.to_datetime('20160101')
        end_date = pd.to_datetime('20180101')
        self.baseline_df = self.df[
            (self.df['trade_date'] >= start_date) &
            (self.df['trade_date'] <= end_date)
            ].copy()

        print(f"基准数据期间: {self.baseline_df['trade_date'].min()} 到 {self.baseline_df['trade_date'].max()}")
        print(f"基准交易日数: {len(self.baseline_df)}")

        # 在基准数据上计算指标
        self.baseline_df['high_low_ratio'] = (self.baseline_df['high'] - self.baseline_df['low']) / self.baseline_df[
            'high']
        self.baseline_df['close_open_volume'] = (self.baseline_df['close'] - self.baseline_df['open']) / \
                                                self.baseline_df['vol']

        # 计算基准数据的分位点
        self.quantiles = {
            'high_low_ratio': self.baseline_df['high_low_ratio'].quantile([0.2, 0.4, 0.6, 0.8]).values,
            'close_open_volume': self.baseline_df['close_open_volume'].quantile([0.2, 0.4, 0.6, 0.8]).values,
            'amount': self.baseline_df['amount'].quantile([0.2, 0.4, 0.6, 0.8]).values
        }

        print("分位点计算完成")

    def get_discrete_data(self, target_date):
        """
        获取指定日期的离散化数据

        参数:
        target_date: 目标日期，格式可以是字符串'YYYYMMDD'或datetime对象

        返回:
        dict: 包含离散化后的指标数据
        """
        # 转换日期格式
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date, format='%Y%m%d')

        # 查找目标日期的数据
        target_data = self.df[self.df['trade_date'] == target_date]

        if target_data.empty:
            return {"error": f"未找到日期 {target_date} 的数据"}

        row = target_data.iloc[0]

        # 计算目标日期的指标
        high_low_ratio = (row['high'] - row['low']) / row['high']
        close_open_volume = (row['close'] - row['open']) / row['vol']
        amount = row['amount']

        # 使用基准分位点进行离散化
        def discretize_value(value, quantiles):
            if value <= quantiles[0]:
                return 1
            elif value <= quantiles[1]:
                return 2
            elif value <= quantiles[2]:
                return 3
            elif value <= quantiles[3]:
                return 4
            else:
                return 5

        result = {
            'high_low_rank': discretize_value(high_low_ratio, self.quantiles['high_low_ratio']),
            'close_open_volume_rank': discretize_value(close_open_volume, self.quantiles['close_open_volume']),
            'amount_rank': discretize_value(amount, self.quantiles['amount']),
        }

        return result

    def get_date_range_data(self, start_date, end_date):
        """
        获取日期范围内的所有离散化数据

        参数:
        start_date, end_date: 开始和结束日期，格式可以是字符串'YYYYMMDD'或datetime对象

        返回:
        DataFrame: 包含离散化后的指标数据
        """
        # 转换日期格式
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date, format='%Y%m%d')
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date, format='%Y%m%d')

        # 获取日期范围内的数据
        date_range_df = self.df[
            (self.df['trade_date'] >= start_date) &
            (self.df['trade_date'] <= end_date)
            ].copy()

        results = []
        for _, row in date_range_df.iterrows():
            result = self.get_discrete_data(row['trade_date'])
            if 'error' not in result:
                results.append({
                    'ts_code': result['ts_code'],
                    'trade_date': result['trade_date'],
                    'high_low_rank': result['high_low_rank'],
                    'close_open_volume_rank': result['close_open_volume_rank'],
                    'amount_rank': result['amount_rank']
                })

        return pd.DataFrame(results)
# 马尔可夫环境下的判断机器
class Agent():
    def __init__(self, account, data_handler, Epsilon, Alpha):
        file_path1 = r"C:\Users\chanpi\Desktop\task\中证500指数_201601-202506.csv"
        self.env = DiscreteIndexEnvironment(file_path1)
        self.data_handler = data_handler
        self.account = account
        self.performance = PerformanceAnalysis(account)
        self.value = np.zeros((3, 3, 3))#因子状态格处置：
        """
        #
        #
        一共27个储存
        3 @ 3 @ 3
        """
        self.Epsilon = Epsilon
        self.Alpha = Alpha
        self.pre_state = np.zeros(3, dtype=int)  # 上一个状态

    def decide(self , state):  #返回动作
        if np.random.binomial(1, self.Epsilon):
            random_bit = np.random.binomial(1, 0.5)
            return random_bit  # 结果要么是 0，要么是 1
        else:
            Value = self.value[tuple(state)]
            return Value >= -0.005



    def receive(self):
        stock_data = self.data_handler.get_stock_data() - 1  # 获取昨天的日期数据
        result = env.get_discrete_data(stock_data)
        state = result = list(result.values())
        self.pre_state = state
# 这里一定要注意等下先换感受再决策（晚上感受，第二天决策）
        return state

    def feedback(self, state):
        stock_data = self.data_handler.get_stock_data()
        index_data = self.data_handler.get_index_price(
            start_date=stock_data - 1,
            end_date=stock_data,
            fields=['date', 'close']
        )
        index_data = index_data.sort_values('date')
        index_return = index_data['close'].iloc[1] / index_data['close'].iloc[0] - 1
        strategy_returns = self.performance.strategy_returns
        feedback = strategy_returns - index_data['return'] / 2
        self.value[tuple(state)] += feedback


# 使用示例
if __name__ == "__main__":
    file_path = r"C:\Users\chanpi\Desktop\task\中证500指数_201601-202506.csv"

    try:
        # 初始化环境
        env = DiscreteIndexEnvironment(file_path)

        # 测试单个日期（2019年的日期）
        test_date = '20190102'  # 2019年的日期
        result = env.get_discrete_data(test_date)

        print(f"\n=== 单个日期测试 ({test_date}) ===")
        if 'error' in result:
            print(f"错误: {result['error']}")
        else:
            print(f"离散化结果:")
            print(f"  high_low_rank: {result['high_low_rank']}")
            print(f"  close_open_volume_rank: {result['close_open_volume_rank']}")
            print(f"  amount_rank: {result['amount_rank']}")
            print(f"\n原始值:")
            print(f"  high_low_ratio: {result['original_high_low_ratio']:.6f}")
            print(f"  close_open_volume: {result['original_close_open_volume']:.10f}")
            print(f"  amount: {result['original_amount']:.2f}")

        # 测试日期范围
        print(f"\n=== 日期范围测试 (2019年1月) ===")
        range_result = env.get_date_range_data('20190101', '20190131')
        print(f"获取到 {len(range_result)} 天的数据")
        if not range_result.empty:
            print("前5天数据:")
            print(range_result.head())

        # 验证分位点的一致性
        print(f"\n=== 分位点验证 ===")
        print("基准分位点 (基于2016-2018数据):")
        for indicator, quantiles in env.quantiles.items():
            print(f"  {indicator}: {quantiles}")

    except FileNotFoundError:
        print(f"文件未找到，请检查文件路径: {file_path}")
    except Exception as e:
        print(f"处理数据时出现错误: {e}")




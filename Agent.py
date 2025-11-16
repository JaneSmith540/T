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
    def __init__(self, account, data_handler, Epsilon=0.1, Alpha=0.1):
        file_path1 = r"C:\Users\chanpi\Desktop\task\中证500指数_201601-202506.csv"
        self.env = DiscreteIndexEnvironment(file_path1)
        self.data_handler = data_handler
        self.account = account
        self.value = np.zeros((5, 5, 5))  # 状态价值函数
        self.Epsilon = Epsilon
        self.Alpha = Alpha
        self.pre_state = None
        self.pre_action = None
        self.current_state = None


        # 学习统计
        self.learning_updates = 0
        self.total_reward = 0.0

    def decide(self, state):
        """决策函数 - 修复决策逻辑"""
        if state is None:
            self.log.warning("状态为None，使用默认决策")
            return 1

        try:
            if np.random.binomial(1, self.Epsilon):
                # 探索：随机选择动作
                action = np.random.randint(0, 2)
                self.log.debug(f"探索决策: 状态{state} -> 动作{action}")
                return action
            else:
                # 利用：根据价值函数选择动作
                state_tuple = tuple(state)
                state_value = self.value[state_tuple]

                # 如果价值相等，随机选择；否则选择价值高的动作
                if state_value == 0:
                    action = np.random.randint(0, 2)
                else:
                    action = 1 if state_value > 0 else 0

                self.log.debug(f"利用决策: 状态{state} -> 价值{state_value:.4f} -> 动作{action}")
                return action
        except Exception as e:
            self.log.error(f"决策错误: {e}")
            return 1

    def receive(self, date):
        """接收环境状态 - 修复状态获取"""
        try:
            # 获取离散化状态
            result = self.env.get_discrete_data(date)
            if 'error' not in result:
                # 确保状态值在有效范围内
                state = [
                    max(0, min(4, result['high_low_rank'] - 1)),
                    max(0, min(4, result['close_open_volume_rank'] - 1)),
                    max(0, min(4, result['amount_rank'] - 1))
                ]
                self.current_state = state
                self.log.info(f"接收状态: 日期{date} -> 状态{state}")
                return state
            else:
                self.log.warning(f"获取状态失败: {result['error']}, 使用中性状态")
                return [2, 2, 2]  # 返回中性状态
        except Exception as e:
            self.log.error(f"接收状态错误: {e}")
            return [2, 2, 2]

    def feedback(self, reward):
        """根据奖励更新价值函数 - 修复学习逻辑"""
        if self.pre_state is not None and self.current_state is not None:
            try:
                # 简单的时序差分学习
                pre_state_tuple = tuple(self.pre_state)
                current_value = self.value[pre_state_tuple]

                # Q-learning 更新规则
                new_value = current_value + self.Alpha * (reward - current_value)
                self.value[pre_state_tuple] = new_value

                self.learning_updates += 1
                self.total_reward += reward

                self.log.info(
                    f"学习更新: 状态{self.pre_state} 价值{current_value:.6f} -> {new_value:.6f} (奖励:{reward:.6f})")

            except Exception as e:
                self.log.error(f"学习更新错误: {e}")
        else:
            self.log.debug("没有先前状态，跳过学习")

        # 更新历史记录
        self.pre_state = self.current_state

    def get_learning_progress(self):
        """获取学习进度"""
        # 使用绝对值阈值来判断是否学习过
        learned_states = np.sum(np.abs(self.value) > 1e-8)
        total_states = self.value.size
        progress = learned_states / total_states
        return progress

    def print_learning_status(self):
        """打印学习状态"""
        progress = self.get_learning_progress()
        learned_states = np.sum(np.abs(self.value) > 1e-8)
        total_states = self.value.size

        print(f"学习进度: {progress:.2%} ({learned_states}/{total_states} 状态已学习)")
        print(f"价值函数范围: [{self.value.min():.6f}, {self.value.max():.6f}]")
        print(f"学习更新次数: {self.learning_updates}")
        print(f"累计奖励: {self.total_reward:.6f}")

        # 打印一些学习示例
        non_zero_indices = np.where(np.abs(self.value) > 1e-8)
        if len(non_zero_indices[0]) > 0:
            print("学习示例:")
            for i in range(min(3, len(non_zero_indices[0]))):
                state = [non_zero_indices[0][i], non_zero_indices[1][i], non_zero_indices[2][i]]
                value = self.value[tuple(state)]
                print(f"  状态{state}: 价值{value:.6f}")


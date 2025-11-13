import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Performance_Analysis import PerformanceAnalysis
from Visualization import BacktestVisualization
from tqdm import tqdm
import logging  # 新增：用于日志输出

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


# 账户类
class Account:
    def __init__(self, initial_cash=100000):
        """初始化账户"""
        self.initial_cash = initial_cash  # 初始资金
        self.cash = initial_cash  # 当前现金
        self.positions = {}  # 持仓情况 {股票代码: 持股数量}
        self.trade_history = []  # 交易历史
        self.total_assets = []  # 每日总资产记录
        self.dates = []  # 日期记录

    def buy(self, date, stock_code, price, amount):
        """买入股票"""
        cost = price * amount
        # 计算手续费（买入佣金万分之三，最低5元）
        commission = 0                   # max(0.0003 * cost, 5)
        total_cost = cost + commission

        if self.cash >= total_cost:
            self.cash -= total_cost
            if stock_code in self.positions:
                self.positions[stock_code] += amount
            else:
                self.positions[stock_code] = amount

            # 记录交易
            self.trade_history.append({
                'date': date,
                'stock_code': stock_code,
                'action': 'buy',
                'price': price,
                'amount': amount,
                'cost': total_cost
            })
            return True
        return False

    def sell(self, date, stock_code, price, amount):
        """卖出股票"""
        if stock_code not in self.positions or self.positions[stock_code] < amount:
            return False

        revenue = price * amount
        # 计算手续费（卖出佣金万分之三+印花税千分之一，最低5元）
        commission = 0                   # max(0.0003 * revenue, 5)
        tax = 0            # 0.001 * revenue
        total_cost = commission + tax

        self.cash += revenue - total_cost
        self.positions[stock_code] -= amount
        if self.positions[stock_code] == 0:
            del self.positions[stock_code]

        # 记录交易
        self.trade_history.append({
            'date': date,
            'stock_code': stock_code,
            'action': 'sell',
            'price': price,
            'amount': amount,
            'revenue': revenue - total_cost
        })
        return True

    def calculate_total_assets(self, date, stock_prices):
        """计算总资产（现金+持仓市值）"""
        position_value = 0
        for stock_code, amount in self.positions.items():
            if stock_code in stock_prices:
                price = stock_prices[stock_code]
                value = price * amount
                position_value += value
                # 新增：打印每日持仓市值明细（调试用）
                log.debug(f"[{date}] 持仓 {stock_code}：价格={price:.2f}，数量={amount}，市值={value:.2f}")
            else:
                log.warning(f"[{date}] 未获取到 {stock_code} 的价格数据，无法计算该股票市值")

        total = self.cash + position_value
        self.total_assets.append(total)
        self.dates.append(date)
        log.info(f"[{date}] 总资产：{total:.2f}（现金：{self.cash:.2f}，持仓市值：{position_value:.2f}）")
        return total


# 回测引擎类
class BacktestEngine:
    def __init__(self, data_handler, strategy_class, initial_cash=100000, max_stock_holdings=None):
        """
        初始化回测引擎
        :param data_handler: 数据处理器
        :param strategy_class: 策略类
        :param initial_cash: 初始资金
        :param max_stock_holdings: 最大持股数量限制（None表示无限制）
        """
        self.data_handler = data_handler
        self.strategy_class = strategy_class
        self.account = Account(initial_cash)
        self.max_stock_holdings = max_stock_holdings  # 最大持股数量限制

        stock_data = self.data_handler.get_stock_data()
        unique_dates = stock_data.index.unique()
        self.dates = pd.DatetimeIndex(unique_dates).sort_values()

        self.benchmark_returns = None
        self.strategy_returns = None

        # 初始化上下文，加入最大持股限制
        self.context = {
            'account': self.account,
            'data_handler': data_handler,
            'current_dt': None,
            'portfolio': {
                'available_cash': self.account.cash,
                'positions': self.account.positions,
                'max_stock_holdings': self.max_stock_holdings
            }
        }

        self.strategy = self.strategy_class(self.context)

    def check_holding_limit(self):
        """检查是否达到最大持股数量限制"""
        if self.max_stock_holdings is None:
            return True  # 无限制时返回True表示可以买入
        # 当前持股数量小于最大限制时返回True
        return len(self.account.positions) < self.max_stock_holdings

    def run(self, start_date=None, end_date=None):
        """运行回测"""
        print(f"原始数据日期范围: {self.dates.min()} 至 {self.dates.max()}")

        if start_date and end_date:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            mask = (self.dates >= start_date) & (self.dates <= end_date)
            trade_dates = self.dates[mask]
            print(f"筛选后日期范围: {start_date} 至 {end_date}")
            print(f"有效交易日数量: {len(trade_dates)}")
        else:
            trade_dates = self.dates

        if len(trade_dates) == 0:
            raise ValueError("没有找到符合条件的交易日期，请检查日期范围是否在数据范围内")

        # 打印最大持股限制信息
        if self.max_stock_holdings:
            print(f"启用最大持股数量限制: {self.max_stock_holdings}只")
        else:
            print("未设置最大持股数量限制")

        self.strategy.initialize()

        print(f"回测开始，开始日期: {trade_dates[0].strftime('%Y-%m-%d')}")
        print(f"结束日期: {trade_dates[-1].strftime('%Y-%m-%d')}")

        # 循环处理每个交易日
        for date in tqdm(trade_dates, desc="回测进度"):
            self.context['current_dt'] = date
            self.context['portfolio']['available_cash'] = self.account.cash
            # 更新当前持股数量到上下文
            self.context['portfolio']['current_holdings_count'] = len(self.account.positions)

            self.strategy.before_market_open(date)
            self.strategy.market_open(date)
            self.strategy.after_market_close(date)

            # 获取当日所有股票的收盘价数据
            daily_stock_data = self.data_handler.get_single_day_data(date)

            # 关键修复：获取当前所有持仓股票的代码
            holding_stocks = self.account.positions.keys()

            # 构建持仓股票的价格字典（仅包含持仓股票的当日收盘价）
            stock_prices = {}
            for stock_code in holding_stocks:
                if stock_code in daily_stock_data.index:
                    stock_prices[stock_code] = daily_stock_data.loc[stock_code]
                else:
                    log.warning(f"[{date}] 当日数据中不存在 {stock_code} 的价格，可能为停牌或未上市")

            # 传递完整的持仓股票价格字典用于总资产计算
            self.account.calculate_total_assets(date, stock_prices)

        print("回测完成!")
        self.performance = PerformanceAnalysis(self.account)

        self.visualization = BacktestVisualization(
            self.account,
            self.performance.strategy_returns
        )

        self.visualization.plot_results()
        self.visualization.print_performance()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Performance_Analysis import PerformanceAnalysis
from Visualization import BacktestVisualization
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


class Account:
    def __init__(self, initial_cash=100000):
        """初始化账户"""
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}
        self.trade_history = []
        self.total_assets = []
        self.dates = []
        self.daily_returns = []  # 新增：记录每日收益率

    def buy(self, date, stock_code, price, amount):
        """买入股票"""
        cost = price * amount
        # 计算手续费（买入佣金万分之三，最低5元）
        commission = 0  # max(0.0003 * cost, 5)
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
            # log.info(f"[{date}] 买入 {stock_code}: {amount}股 @ {price:.2f}, 成本{total_cost:.2f}")
            return True
        else:
            log.warning(f"[{date}] 买入失败: 资金不足 {self.cash:.2f} < {total_cost:.2f}")
            return False

    def sell(self, date, stock_code, price, amount):
        """卖出股票"""
        if stock_code not in self.positions or self.positions[stock_code] < amount:
            log.warning(f"[{date}] 卖出失败: 持仓不足 {stock_code}")
            return False

        revenue = price * amount
        # 计算手续费（卖出佣金万分之三+印花税千分之一，最低5元）
        commission = 0  # max(0.0003 * revenue, 5)
        tax = 0  # 0.001 * revenue
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
        # log.info(f"[{date}] 卖出 {stock_code}: {amount}股 @ {price:.2f}, 收入{revenue - total_cost:.2f}")
        return True

    def calculate_total_assets(self, date, stock_prices):
        """计算总资产（现金+持仓市值）"""
        position_value = 0
        position_details = []

        for stock_code, amount in self.positions.items():
            if stock_code in stock_prices:
                price = stock_prices[stock_code]
                value = price * amount
                position_value += value
                position_details.append(f"{stock_code}:{amount}×{price:.2f}={value:.2f}")
            else:
                log.warning(f"[{date}] 未获取到 {stock_code} 的价格数据，无法计算该股票市值")

        total = self.cash + position_value
        self.total_assets.append(total)
        self.dates.append(date)

        # 计算日收益率
        if len(self.total_assets) > 1:
            prev_assets = self.total_assets[-2]
            daily_return = (total - prev_assets) / prev_assets
            self.daily_returns.append(daily_return)
        else:
            self.daily_returns.append(0.0)

        log.info(f"[{date}] 总资产: {total:.2f} (现金: {self.cash:.2f}, 持仓: {position_value:.2f})")
        if position_details:
            log.debug(f"[{date}] 持仓明细: {', '.join(position_details)}")

        return total

    def get_current_assets(self):
        """获取当前总资产"""
        return self.total_assets[-1] if self.total_assets else self.initial_cash


class BacktestEngine:
    def __init__(self, data_handler, strategy_class, initial_cash=100000, max_stock_holdings=None):
        """
        初始化回测引擎
        :param data_handler: 数据处理器
        :param strategy_class: 策略类
        :param initial_cash: 初始资金
        :param max_stock_holdings: 最大持股数量限制
        """
        self.data_handler = data_handler
        self.strategy_class = strategy_class
        self.account = Account(initial_cash)
        self.max_stock_holdings = max_stock_holdings

        # 获取交易日期
        stock_data = self.data_handler.get_stock_data()
        unique_dates = stock_data.index.unique()
        self.dates = pd.DatetimeIndex(unique_dates).sort_values()

        self.benchmark_returns = None
        self.strategy_returns = None

        # 初始化上下文
        self.context = {
            'account': self.account,
            'data_handler': data_handler,
            'current_dt': None,
            'portfolio': {
                'available_cash': self.account.cash,
                'positions': self.account.positions,
                'max_stock_holdings': self.max_stock_holdings,
                'current_holdings_count': 0
            }
        }

        self.strategy = self.strategy_class(self.context)
        self.performance = None
        self.visualization = None

    def check_holding_limit(self):
        """检查是否达到最大持股数量限制"""
        if self.max_stock_holdings is None:
            return True
        return len(self.account.positions) < self.max_stock_holdings

    def _get_daily_stock_prices(self, date):
        """获取当日所有持仓股票的价格"""
        try:
            # 获取当日所有股票的收盘价数据
            daily_stock_data = self.data_handler.get_single_day_data(date)

            # 构建持仓股票的价格字典
            stock_prices = {}
            for stock_code in self.account.positions.keys():
                if stock_code in daily_stock_data.index:
                    stock_prices[stock_code] = daily_stock_data.loc[stock_code]
                else:
                    # 如果当日没有数据，尝试使用最近的价格
                    try:
                        recent_data = self.data_handler.get_price(
                            stock_code,
                            end_date=date,
                            count=1,
                            fields=['close']
                        )
                        if not recent_data.empty:
                            stock_prices[stock_code] = recent_data['close'].iloc[-1]
                        else:
                            log.warning(f"[{date}] 无法获取 {stock_code} 的价格，使用0计算")
                            stock_prices[stock_code] = 0
                    except Exception as e:
                        log.warning(f"[{date}] 获取 {stock_code} 价格失败: {e}")
                        stock_prices[stock_code] = 0

            return stock_prices

        except Exception as e:
            log.error(f"[{date}] 获取股票价格失败: {e}")
            return {}

    def run(self, start_date=None, end_date=None):
        """运行回测"""
        log.info("开始回测...")
        print(f"原始数据日期范围: {self.dates.min()} 至 {self.dates.max()}")

        # 筛选交易日期
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

        # 打印配置信息
        print(f"初始资金: {self.account.initial_cash:,.2f}")
        if self.max_stock_holdings:
            print(f"最大持股数量限制: {self.max_stock_holdings}只")
        else:
            print("未设置最大持股数量限制")

        # 初始化策略
        self.strategy.initialize()
        print(f"回测开始日期: {trade_dates[0].strftime('%Y-%m-%d')}")
        print(f"回测结束日期: {trade_dates[-1].strftime('%Y-%m-%d')}")

        # 主回测循环
        for i, date in enumerate(tqdm(trade_dates, desc="回测进度")):
            log.info(f"\n=== 交易日 {i + 1}/{len(trade_dates)}: {date.strftime('%Y-%m-%d')} ===")

            # 更新上下文
            self.context['current_dt'] = date
            self.context['portfolio']['available_cash'] = self.account.cash
            self.context['portfolio']['current_holdings_count'] = len(self.account.positions)

            try:
                # 1. 开盘前：Agent接收状态并决策
                self.strategy.before_market_open(date)

                # 2. 开盘时：执行交易
                self.strategy.market_open(date)

                # 3. 获取当日股票价格并计算资产
                stock_prices = self._get_daily_stock_prices(date)
                current_assets = self.account.calculate_total_assets(date, stock_prices)

                # 4. 收盘后：记录数据用于学习
                self.strategy.after_market_close(date)

                # 打印当日总结
                log.info(f"[{date}] 当日总结: 总资产={current_assets:,.2f}, 现金={self.account.cash:,.2f}, "
                         f"持仓数量={len(self.account.positions)}, 当日收益率={self.account.daily_returns[-1]:.4f}")

            except Exception as e:
                log.error(f"[{date}] 回测执行错误: {e}")
                continue

        print("回测完成!")

        # 性能分析
        self._perform_analysis()

        # 可视化结果
        self._visualize_results()

        # 打印学习总结
        self._print_learning_summary()

    def _perform_analysis(self):
        """执行性能分析"""
        try:
            self.performance = PerformanceAnalysis(self.account)
            print("\n=== 回测性能分析 ===")
            print(f"最终资产: {self.account.get_current_assets():,.2f}")
            print(f"总收益率: {self.performance.total_return:.2%}")
            print(f"年化收益率: {self.performance.annual_return:.2%}")
            print(f"最大回撤: {self.performance.max_drawdown:.2%}")
            print(f"夏普比率: {self.performance.sharpe_ratio:.2f}")
        except Exception as e:
            log.error(f"性能分析失败: {e}")

    def _visualize_results(self):
        """可视化回测结果"""
        try:
            self.visualization = BacktestVisualization(
                self.account,
                self.performance.strategy_returns if self.performance else []
            )
            self.visualization.plot_results()
            self.visualization.print_performance()
        except Exception as e:
            log.error(f"可视化失败: {e}")

    def _print_learning_summary(self):
        """打印学习总结"""
        try:
            if hasattr(self.strategy, 'print_learning_summary'):
                print("\n=== Agent学习总结 ===")
                self.strategy.print_learning_summary()
            else:
                print("\n=== 策略执行完成 ===")
                print("注意: 策略未提供学习总结功能")
        except Exception as e:
            log.error(f"打印学习总结失败: {e}")

    def get_trade_history(self):
        """获取交易历史"""
        return pd.DataFrame(self.account.trade_history)

    def get_portfolio_history(self):
        """获取投资组合历史"""
        return pd.DataFrame({
            'date': self.account.dates,
            'total_assets': self.account.total_assets,
            'daily_returns': self.account.daily_returns
        })

    def save_results(self, file_path):
        """保存回测结果到文件"""
        try:
            # 保存交易历史
            trade_df = self.get_trade_history()
            trade_df.to_csv(f"{file_path}_trades.csv", index=False)

            # 保存资产历史
            portfolio_df = self.get_portfolio_history()
            portfolio_df.to_csv(f"{file_path}_portfolio.csv", index=False)

            log.info(f"回测结果已保存到: {file_path}_*.csv")
        except Exception as e:
            log.error(f"保存结果失败: {e}")
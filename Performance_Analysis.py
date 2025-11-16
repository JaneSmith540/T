import pandas as pd
import numpy as np


class PerformanceAnalysis:
    def __init__(self, account):
        self.account = account
        self.strategy_returns = None
        self.cumulative_returns = None
        self.cumulative_net_assets = None

        # 只有在有资产数据时才计算收益率
        if len(self.account.total_assets) > 0:
            self.calculate_returns()
            self.calculate_cumulative_returns()

    def calculate_returns(self):
        """计算策略收益率"""
        if len(self.account.total_assets) < 2:
            self.strategy_returns = pd.Series(dtype='float64')
            return

        # 使用总资产计算日收益率
        self.strategy_returns = pd.Series(
            self.account.total_assets,
            index=self.account.dates
        ).pct_change().fillna(0)

    def calculate_cumulative_returns(self):
        """计算累计收益率"""
        if self.strategy_returns is None:
            self.calculate_returns()

        if not self.strategy_returns.empty:
            self.cumulative_returns = (1 + self.strategy_returns).cumprod() - 1
            self.cumulative_net_assets = pd.Series(
                self.account.total_assets,
                index=self.account.dates
            )

    def get_total_return(self):
        """计算总收益率"""
        if self.cumulative_returns is None:
            self.calculate_cumulative_returns()

        if self.cumulative_returns.empty:
            return 0.0
        return self.cumulative_returns.iloc[-1] * 100  # 转换为百分比

    def get_annualized_return(self):
        """计算年化收益率"""
        if len(self.account.dates) < 2:
            return 0.0

        total_days = (self.account.dates[-1] - self.account.dates[0]).days
        if total_days == 0:
            return 0.0

        annual_factor = 365 / total_days
        total_return = self.get_total_return() / 100  # 转换为小数
        return (pow(1 + total_return, annual_factor) - 1) * 100

    def get_sharpe_ratio(self, risk_free_rate=0):
        """计算夏普比率"""
        if self.strategy_returns is None:
            self.calculate_returns()

        if self.strategy_returns.empty:
            return 0.0

        excess_returns = self.strategy_returns - risk_free_rate / 252  # 假设252个交易日
        return excess_returns.mean() / excess_returns.std() * (252 ** 0.5)

    def get_max_drawdown(self):
        """计算最大回撤"""
        if self.cumulative_net_assets is None:
            self.calculate_cumulative_returns()

        if len(self.cumulative_net_assets) == 0:
            return 0.0

        # 使用累计净值计算回撤
        cumulative_values = self.cumulative_net_assets.values
        peak = cumulative_values[0]
        max_drawdown = 0

        for value in cumulative_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        return max_drawdown * 100  # 转换为百分比

    def get_trade_count(self):
        """获取总交易次数"""
        return len(self.account.trade_history)

    def get_buy_sell_count(self):
        """获取买入和卖出次数"""
        buy_count = sum(1 for trade in self.account.trade_history if trade['action'] == 'buy')
        sell_count = sum(1 for trade in self.account.trade_history if trade['action'] == 'sell')
        return buy_count, sell_count

    def get_win_rate(self):
        """计算胜率"""
        if not self.account.trade_history:
            return 0.0

        trades = pd.DataFrame(self.account.trade_history)
        sell_trades = trades[trades['action'] == 'sell']

        if sell_trades.empty:
            return 0.0

        if 'profit' in sell_trades.columns:
            winning_trades = len(sell_trades[sell_trades['profit'] > 0])
            return (winning_trades / len(sell_trades)) * 100
        else:
            return 0.0

    def get_avg_trade_return(self):
        """计算平均交易收益率"""
        if not self.account.trade_history:
            return 0.0

        trades = pd.DataFrame(self.account.trade_history)
        if 'return_rate' in trades.columns:
            return trades['return_rate'].mean() * 100
        else:
            return 0.0

    def validate_data(self):
        """验证数据完整性"""
        issues = []

        # 检查资产数据
        if len(self.account.total_assets) == 0:
            issues.append("没有资产数据")

        # 检查资产值是否合理
        if any(asset <= 0 for asset in self.account.total_assets):
            issues.append("存在非正资产值")

        # 检查日期数据
        if len(self.account.dates) != len(self.account.total_assets):
            issues.append("日期和资产数据长度不匹配")

        return issues

    def get_performance_summary(self):
        """生成完整的绩效摘要"""
        # 首先验证数据
        data_issues = self.validate_data()
        if data_issues:
            print(f"数据警告: {', '.join(data_issues)}")

        total_return = self.get_total_return()
        annual_return = self.get_annualized_return()
        sharpe_ratio = self.get_sharpe_ratio()
        max_drawdown = self.get_max_drawdown()
        volatility = self.get_volatility()
        calmar_ratio = self.get_calmar_ratio()
        trade_count = self.get_trade_count()
        buy_count, sell_count = self.get_buy_sell_count()
        win_rate = self.get_win_rate()
        avg_trade_return = self.get_avg_trade_return()

        # 调试信息
        print(f"调试信息:")
        print(f"  初始资产: {self.account.total_assets[0] if self.account.total_assets else 'N/A'}")
        print(f"  最终资产: {self.account.total_assets[-1] if self.account.total_assets else 'N/A'}")
        print(f"  交易日数: {len(self.account.dates)}")
        print(f"  收益率序列长度: {len(self.strategy_returns) if self.strategy_returns is not None else 0}")

        summary = {
            '总收益率 (%)': round(total_return, 2),
            '年化收益率 (%)': round(annual_return, 2),
            '夏普比率': round(sharpe_ratio, 3),
            '最大回撤 (%)': round(max_drawdown, 2),
            '年化波动率 (%)': round(volatility, 2),
            'Calmar比率': round(calmar_ratio, 3),
            '总交易次数': trade_count,
            '买入次数': buy_count,
            '卖出次数': sell_count,
            '胜率 (%)': round(win_rate, 2),
            '平均交易收益率 (%)': round(avg_trade_return, 2)
        }

        return summary
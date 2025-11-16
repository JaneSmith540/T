# 修改后的Strategy_Core.py
from Utilities import log
import pandas as pd
import numpy as np
from Data_Handling import get_weight, get_price, get_index_price
from Agent import Agent  # 导入Agent类


class WeightBasedStrategy:
    def __init__(self, context):
        self.context = context
        self.g = type('Global', (object,), {})()  # 模拟全局变量
        self.g.securities = []  # 中证500成分股
        self.g.weights = {}  # 股票权重
        self.g.is_initial_purchase_done = False  # 初始半仓标记
        self.g.initial_half_pos = {}  # 初始半仓持仓 {股票代码: 数量}

        # 初始化Agent
        self.agent = Agent(
            account=context['account'],
            data_handler=context['data_handler'],
            Epsilon=0.1,
            Alpha=0.05
        )

        # 学习相关变量
        self.g.current_state = [0, 0, 0]  # Agent的初始状态
        self.g.last_date = None
        self.g.last_assets = None
        self.g.last_benchmark_price = None

    def initialize(self):
        """初始化策略，先建立半仓底仓"""
        log.info('策略初始化：建立半仓底仓')
        try:
            weight_df = get_weight()
            self.g.securities = weight_df['ts_code'].unique().tolist()
            self.g.weights = dict(zip(weight_df['ts_code'], weight_df['weight']))
            log.info(f"股票池包含 {len(self.g.securities)} 只中证500成分股")

            # 初始化学习记录
            self.g.last_assets = self.context['account'].initial_cash
            self.g.last_date = None
            self.g.last_benchmark_price = None

        except Exception as e:
            log.error(f"初始化失败：{str(e)}")

    def before_market_open(self, date):
        """开盘前决策：根据Agent输出确定当日策略"""
        try:
            # 如果是第二天及以后，计算前一天的奖励并学习
            if self.g.last_date is not None:
                self._learn_from_previous_day()

            # 获取当前市场状态
            state = self.agent.receive(date)
            self.g.current_state = state

            # Agent决策
            self.g.agent_decision = self.agent.decide(state)
            log.info(f"[{date}] Agent状态: {state}, 决策: {self.g.agent_decision} (1=看多做T, 0=看空做T)")

        except Exception as e:
            log.error(f"开盘前决策错误: {e}")
            self.g.agent_decision = 1  # 默认看多

    def market_open(self, date):
        """开盘时操作"""
        account = self.context['account']

        # 第一步：建立初始半仓（仅首次运行）
        if not self.g.is_initial_purchase_done:
            self._initial_half_position(date)
            self.g.is_initial_purchase_done = True
            return

        # 第二步：根据Agent决策执行当日操作
        if self.g.agent_decision == 1:
            # 看多策略：开盘再买半仓（总仓位100%）
            self._open_buy_half(date)
        else:
            # 看空策略：开盘卖出半仓（总仓位0%）
            self._open_sell_half(date)

    def after_market_close(self, date):
        """收盘时操作"""
        account = self.context['account']

        if not self.g.is_initial_purchase_done:
            return

        if self.g.agent_decision == 1:
            # 看多策略：未达盈利1%则收盘卖出半仓（回到50%仓位）
            self._close_sell_half_if_no_profit(date)
        else:
            # 看空策略：指数未跌1%则收盘买入半仓（回到50%仓位）
            self._close_buy_half_if_no_drop(date)

        # 记录当日收盘数据用于学习
        self._record_daily_data(date)

    def _record_daily_data(self, date):
        """记录当日数据用于学习"""
        try:
            account = self.context['account']
            current_assets = account.total_assets[-1] if account.total_assets else account.initial_cash

            # 获取当日基准收盘价
            benchmark_price = self._get_benchmark_close_price(date)

            # 记录数据
            self.g.last_assets = current_assets
            self.g.last_date = date
            self.g.last_benchmark_price = benchmark_price

            log.info(f"[{date}] 记录数据: 总资产={current_assets:.2f}, 基准收盘价={benchmark_price:.2f}")

        except Exception as e:
            log.error(f"记录数据错误: {e}")

    def _learn_from_previous_day(self):
        """从前一日的表现中学习"""
        try:
            if self.g.last_date is None or self.g.last_assets is None:
                return

            account = self.context['account']
            current_assets = account.total_assets[-1] if account.total_assets else account.initial_cash

            # 计算昨日策略收益率
            daily_return = (current_assets - self.g.last_assets) / self.g.last_assets

            # 计算基准收益率
            current_benchmark_price = self._get_benchmark_close_price(self.g.last_date)
            if self.g.last_benchmark_price and current_benchmark_price:
                benchmark_return = (current_benchmark_price - self.g.last_benchmark_price) / self.g.last_benchmark_price
            else:
                benchmark_return = 0.0

            # 计算超额收益作为奖励
            reward = daily_return - benchmark_return

            # 给Agent反馈
            self.agent.feedback(reward)

            log.info(
                f"[{self.g.last_date}] 学习结果: 策略收益={daily_return:.4f}, 基准收益={benchmark_return:.4f}, 奖励={reward:.4f}")

            # 打印学习进度
            if hasattr(self.agent, 'get_learning_progress'):
                progress = self.agent.get_learning_progress()
                log.info(f"学习进度: {progress:.2%}")

        except Exception as e:
            log.error(f"学习过程错误: {e}")

    def _get_benchmark_close_price(self, date):
        """获取基准指数收盘价"""
        try:
            # 使用Data_Handling中的get_index_price方法
            index_data = get_index_price(
                start_date=date,
                end_date=date,
                fields=['date', 'close']
            )
            if not index_data.empty and 'close' in index_data.columns:
                return index_data['close'].iloc[0]
            else:
                log.warning(f"无法获取 {date} 的基准收盘价")
                return None
        except Exception as e:
            log.error(f"获取基准收盘价错误: {e}")
            return None

    def _initial_half_position(self, date):
        """建立初始半仓（50%仓位）"""
        account = self.context['account']
        total_cash = account.initial_cash * 0.5  # 仅用一半资金
        total_weight = sum(self.g.weights.values())

        if total_weight <= 0:
            log.error("权重总和无效，无法建立底仓")
            return

        for security in self.g.securities:
            weight = self.g.weights.get(security, 0)
            if weight <= 0:
                continue

            allocation_ratio = weight / total_weight
            target_value = total_cash * allocation_ratio
            current_price = self._get_current_price(security, date)
            if not current_price:
                continue

            buy_amount = self.calculate_buy_amount(target_value, current_price)
            if buy_amount <= 0:
                continue

            if account.buy(date, security, current_price, buy_amount):
                self.g.initial_half_pos[security] = buy_amount
                log.info(f"初始半仓买入 {security}: {buy_amount}股 @ {current_price:.2f}")

    def _open_buy_half(self, date):
        """开盘买入半仓（基于初始半仓的同等金额）"""
        account = self.context['account']
        total_buy_value = 0
        successful_buys = 0

        for security, initial_amount in self.g.initial_half_pos.items():
            current_price = self._get_current_price(security, date)
            if not current_price:
                continue

            # 买入与初始半仓同等价值的股份
            target_value = initial_amount * current_price
            buy_amount = self.calculate_buy_amount(target_value, current_price)
            if buy_amount <= 0:
                continue

            if account.buy(date, security, current_price, buy_amount):
                total_buy_value += target_value
                successful_buys += 1
                log.info(f"看多策略买入 {security}: {buy_amount}股 @ {current_price:.2f}")

        if successful_buys > 0:
            log.info(f"看多策略完成: 成功买入{successful_buys}只股票, 总价值{total_buy_value:.2f}")

    def _open_sell_half(self, date):
        """开盘卖出全部初始半仓"""
        account = self.context['account']
        total_sell_value = 0
        successful_sells = 0

        for security, amount in self.g.initial_half_pos.items():
            if security in account.positions and account.positions[security] >= amount:
                current_price = self._get_current_price(security, date)
                if not current_price:
                    continue
                if account.sell(date, security, current_price, amount):
                    total_sell_value += amount * current_price
                    successful_sells += 1
                    log.info(f"看空策略卖出 {security}: {amount}股 @ {current_price:.2f}")

        if successful_sells > 0:
            log.info(f"看空策略完成: 成功卖出{successful_sells}只股票, 总价值{total_sell_value:.2f}")

    def _close_sell_half_if_no_profit(self, date):
        """收盘时，若未盈利1%则卖出当日新增仓位"""
        account = self.context['account']
        total_sell_value = 0
        successful_sells = 0

        for security, initial_amount in self.g.initial_half_pos.items():
            current_hold = account.positions.get(security, 0)
            if current_hold <= initial_amount:  # 已达目标仓位
                continue

            # 计算当日涨跌幅
            open_price = self._get_open_price(security, date)
            current_price = self._get_current_price(security, date)
            if not open_price or not current_price:
                continue

            price_change = (current_price - open_price) / open_price
            if price_change < 0.01:  # 盈利未达1%
                sell_amount = current_hold - initial_amount
                if account.sell(date, security, current_price, sell_amount):
                    total_sell_value += sell_amount * current_price
                    successful_sells += 1
                    log.info(
                        f"收盘卖出 {security}: {sell_amount}股 @ {current_price:.2f} (未达盈利1%, 涨幅={price_change:.2%})")

        if successful_sells > 0:
            log.info(f"收盘卖出完成: 成功卖出{successful_sells}只股票, 总价值{total_sell_value:.2f}")

    def _close_buy_half_if_no_drop(self, date):
        """收盘时，若指数未跌1%则买入半仓"""
        index_drop = self._get_index_drop(date)
        if index_drop < 0.01:  # 指数跌幅未达1%
            log.info(f"指数跌幅{index_drop:.2%}未达1%，执行收盘买入")
            self._open_buy_half(date)
        else:
            log.info(f"指数跌幅{index_drop:.2%}达到1%，不执行收盘买入")

    def _get_index_drop(self, date):
        """获取当日指数跌幅"""
        try:
            # 获取当日指数开盘价和收盘价
            index_data = get_index_price(
                start_date=date,
                end_date=date,
                fields=['date', 'open', 'close']
            )
            if len(index_data) == 0:
                return 0
            open_price = index_data['open'].iloc[0]
            close_price = index_data['close'].iloc[0]
            drop_ratio = (open_price - close_price) / open_price
            return max(0, drop_ratio)  # 确保非负
        except Exception as e:
            log.error(f"获取指数跌幅失败: {e}")
            return 0

    def _get_current_price(self, security, date):
        """获取股票当前价格（收盘价）"""
        try:
            data = get_price(security, count=1, fields=['close'], end_date=date)
            return data['close'].iloc[-1] if len(data) > 0 else None
        except Exception as e:
            log.error(f"获取 {security} 价格失败: {e}")
            return None

    def _get_open_price(self, security, date):
        """获取股票开盘价"""
        try:
            data = get_price(security, count=1, fields=['open'], end_date=date)
            return data['open'].iloc[-1] if len(data) > 0 else None
        except Exception as e:
            log.error(f"获取 {security} 开盘价失败: {e}")
            return None

    def calculate_buy_amount(self, target_value, price):
        """计算可买入数量（考虑手续费）"""
        if price <= 0 or target_value <= 0:
            return 0
        max_amount = int(target_value / price)
        if max_amount == 0:
            return 0
        # 计算手续费（买入佣金万分之三，最低5元）
        cost = price * max_amount
        commission = max(0.0003 * cost, 5)
        total_cost = cost + commission
        # 确保总成本不超过目标金额
        while total_cost > target_value and max_amount > 0:
            max_amount -= 1
            cost = price * max_amount
            commission = max(0.0003 * cost, 5)
            total_cost = cost + commission
        return max_amount

    def print_learning_summary(self):
        """打印学习总结"""
        if hasattr(self.agent, 'print_learning_status'):
            self.agent.print_learning_status()
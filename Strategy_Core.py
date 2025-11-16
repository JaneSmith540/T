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
        self.g.initial_prices = {}  # 初始买入价格

        # 初始化Agent
        self.agent = Agent(
            account=context['account'],
            data_handler=context['data_handler'],
            Epsilon=0.1,
            Alpha=0.1
        )

        # 学习相关变量
        self.g.current_state = [0, 0, 0]
        self.g.last_date = None
        self.g.last_assets = None
        self.g.last_benchmark_price = None
        self.g.last_state = None

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
            self.g.last_state = None

        except Exception as e:
            log.error(f"初始化失败：{str(e)}")

    def before_market_open(self, date):
        """开盘前决策：根据Agent输出确定当日策略"""
        try:
            # 获取当前市场状态
            state = self.agent.receive(date)
            self.g.current_state = state

            # 如果是第二天及以后，计算前一天的奖励并学习
            if self.g.last_date is not None and self.g.last_state is not None:
                self._learn_from_previous_day()

            # Agent决策（1=看多做T，0=看空做T）
            self.g.agent_decision = self.agent.decide(state)
            log.info(f"Agent决策：状态{state} -> 决策{self.g.agent_decision}（1=看多做T，0=看空做T）")

            # 记录当前状态用于明天学习
            self.g.last_state = state

        except Exception as e:
            log.error(f"开盘前决策错误：{e}")
            self.g.agent_decision = 1  # 默认看多

    def market_open(self, date):
        """开盘时操作 - 使用开盘价"""
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
        """收盘时操作 - 使用收盘价"""
        account = self.context['account']

        if not self.g.is_initial_purchase_done:
            return

        if self.g.agent_decision == 1:
            # 看多策略：根据指数表现决定卖出价格
            self._close_sell_by_index_performance(date)
        else:
            # 看空策略：根据指数表现决定买入价格
            self._close_buy_by_index_performance(date)

        # 记录当日数据用于学习
        self._record_daily_data(date)

        # 打印当日账户状态
        self._print_account_status(date)

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

            log.info(f"记录学习数据: 日期{date}, 总资产={current_assets:.2f}, 基准价={benchmark_price}")

        except Exception as e:
            log.error(f"记录数据错误: {e}")

    def _learn_from_previous_day(self):
        """从前一日的表现中学习"""
        try:
            if self.g.last_date is None or self.g.last_assets is None:
                log.debug("没有历史数据，跳过学习")
                return

            account = self.context['account']
            current_assets = account.total_assets[-1] if account.total_assets else account.initial_cash

            # 计算昨日策略收益率
            if self.g.last_assets > 0:
                daily_return = (current_assets - self.g.last_assets) / self.g.last_assets
            else:
                daily_return = 0.0

            # 计算基准收益率
            current_benchmark_price = self._get_benchmark_close_price(self.g.last_date)
            if (self.g.last_benchmark_price and current_benchmark_price and
                    self.g.last_benchmark_price > 0):
                benchmark_return = (current_benchmark_price - self.g.last_benchmark_price) / self.g.last_benchmark_price
            else:
                benchmark_return = 0.0

            # 计算超额收益作为奖励
            reward = daily_return - benchmark_return

            # 放大奖励信号（乘以10让学习更明显）
            amplified_reward = reward * 10

            log.info(f"学习计算: 日期={self.g.last_date}, 策略收益={daily_return:.4f}, "
                     f"基准收益={benchmark_return:.4f}, 原始奖励={reward:.4f}, 放大奖励={amplified_reward:.4f}")

            # 给Agent反馈
            self.agent.feedback(amplified_reward)

        except Exception as e:
            log.error(f"学习过程错误: {e}")

    def _get_benchmark_close_price(self, date):
        """获取基准指数收盘价"""
        try:
            # 使用回测引擎提供的指数数据
            if 'index_data' in self.context and self.context['index_data']:
                return self.context['index_data'].get('close')

            # 备用方法
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

    def _get_index_performance(self, date):
        """获取指数当日表现（最高价涨幅和最低价跌幅）"""
        try:
            # 使用回测引擎提供的指数数据
            if 'index_data' in self.context and self.context['index_data']:
                index_data = self.context['index_data']
                open_price = index_data.get('open', 0)
                high_price = index_data.get('high', 0)
                low_price = index_data.get('low', 0)
            else:
                # 备用方法
                index_data = get_index_price(
                    start_date=date,
                    end_date=date,
                    fields=['date', 'open', 'high', 'low', 'close']
                )
                if index_data.empty:
                    return 0, 0
                open_price = index_data['open'].iloc[0]
                high_price = index_data['high'].iloc[0]
                low_price = index_data['low'].iloc[0]

            high_increase = (high_price - open_price) / open_price if open_price > 0 else 0
            low_decrease = (open_price - low_price) / open_price if open_price > 0 else 0

            return high_increase, low_decrease
        except Exception as e:
            log.error(f"获取指数表现失败: {e}")
            return 0, 0

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
            current_price = self._get_open_price(security, date)  # 使用开盘价
            if not current_price:
                continue

            buy_amount = self.calculate_buy_amount(target_value, current_price)
            if buy_amount <= 0:
                continue

            if account.buy(date, security, current_price, buy_amount):
                self.g.initial_half_pos[security] = buy_amount
                self.g.initial_prices[security] = current_price  # 记录初始买入价格
                log.info(f"初始半仓买入 {security}：{buy_amount}股 @ {current_price:.2f}")

    def _open_buy_half(self, date):
        """开盘买入半仓（基于初始半仓的同等金额）- 使用开盘价"""
        account = self.context['account']
        total_buy_value = 0
        successful_buys = 0

        for security, initial_amount in self.g.initial_half_pos.items():
            current_price = self._get_open_price(security, date)  # 使用开盘价
            if not current_price:
                continue

            # 买入与初始半仓同等价值的股份
            target_value = initial_amount * self.g.initial_prices.get(security, current_price)
            buy_amount = self.calculate_buy_amount(target_value, current_price)
            if buy_amount <= 0:
                continue

            if account.buy(date, security, current_price, buy_amount):
                total_buy_value += target_value
                successful_buys += 1
                log.info(f"看多策略开盘买入 {security}：{buy_amount}股 @ {current_price:.2f}")

        if successful_buys > 0:
            log.info(f"看多策略开盘买入完成: 成功买入{successful_buys}只股票, 总价值{total_buy_value:.2f}")

    def _open_sell_half(self, date):
        """开盘卖出全部初始半仓 - 使用开盘价"""
        account = self.context['account']
        total_sell_value = 0
        successful_sells = 0

        for security, amount in self.g.initial_half_pos.items():
            if security in account.positions and account.positions[security] >= amount:
                current_price = self._get_open_price(security, date)  # 使用开盘价
                if not current_price:
                    continue
                if account.sell(date, security, current_price, amount):
                    total_sell_value += amount * current_price
                    successful_sells += 1
                    log.info(f"看空策略开盘卖出 {security}：{amount}股 @ {current_price:.2f}")

        if successful_sells > 0:
            log.info(f"看空策略开盘卖出完成: 成功卖出{successful_sells}只股票, 总价值{total_sell_value:.2f}")

    def _close_sell_by_index_performance(self, date):
        """收盘时根据指数表现决定卖出价格 - 使用收盘价"""
        account = self.context['account']

        # 获取指数表现
        high_increase, low_decrease = self._get_index_performance(date)
        log.info(f"指数表现: 最高涨幅={high_increase:.2%}, 最低跌幅={low_decrease:.2%}")

        total_sell_value = 0
        successful_sells = 0

        for security, initial_amount in self.g.initial_half_pos.items():
            current_hold = account.positions.get(security, 0)
            if current_hold <= initial_amount:  # 已达目标仓位
                continue

            # 如果指数最高涨幅超过1%，使用成本价*1.01作为卖出价，否则使用收盘价
            if high_increase >= 0.01:
                # 使用成本价的1.01倍作为卖出价
                cost_price = self.g.initial_prices.get(security, 0)
                target_sell_price = cost_price * 1.01
                log.info(f"指数涨幅达{high_increase:.2%}，使用目标卖出价: {target_sell_price:.2f}")
            else:
                # 使用收盘价
                target_sell_price = self._get_current_price(security, date)
                log.info(f"指数涨幅{high_increase:.2%}未达1%，使用收盘价: {target_sell_price:.2f}")

            if not target_sell_price or target_sell_price <= 0:
                continue

            sell_amount = current_hold - initial_amount
            if account.sell(date, security, target_sell_price, sell_amount):
                total_sell_value += sell_amount * target_sell_price
                successful_sells += 1
                log.info(f"收盘卖出 {security}：{sell_amount}股 @ {target_sell_price:.2f}")

        if successful_sells > 0:
            log.info(f"收盘卖出完成: 成功卖出{successful_sells}只股票, 总价值{total_sell_value:.2f}")

    def _close_buy_by_index_performance(self, date):
        """收盘时根据指数表现决定买入价格 - 使用收盘价"""
        account = self.context['account']

        # 获取指数表现
        high_increase, low_decrease = self._get_index_performance(date)
        log.info(f"指数表现: 最高涨幅={high_increase:.2%}, 最低跌幅={low_decrease:.2%}")

        total_buy_value = 0
        successful_buys = 0

        # 无论指数跌幅是否达到1%，都执行买入，只是价格不同
        for security, initial_amount in self.g.initial_half_pos.items():
            # 如果指数跌幅超过1%，使用成本价*0.99作为买入价，否则使用收盘价
            if low_decrease >= 0.01:
                cost_price = self.g.initial_prices.get(security, 0)
                target_buy_price = cost_price * 0.99
                log.info(f"指数跌幅达{low_decrease:.2%}，使用目标买入价: {target_buy_price:.2f}")
            else:
                target_buy_price = self._get_current_price(security, date)
                log.info(f"指数跌幅{low_decrease:.2%}未达1%，使用收盘价: {target_buy_price:.2f}")

            if not target_buy_price or target_buy_price <= 0:
                continue

            # 买入与初始半仓同等价值的股份
            target_value = initial_amount * self.g.initial_prices.get(security, target_buy_price)
            buy_amount = self.calculate_buy_amount(target_value, target_buy_price)
            if buy_amount <= 0:
                continue

            if account.buy(date, security, target_buy_price, buy_amount):
                total_buy_value += target_value
                successful_buys += 1
                log.info(f"收盘买入 {security}：{buy_amount}股 @ {target_buy_price:.2f}")

        if successful_buys > 0:
            log.info(f"收盘买入完成: 成功买入{successful_buys}只股票, 总价值{total_buy_value:.2f}")

    def _get_current_price(self, security, date):
        """获取股票当前价格（收盘价）"""
        try:
            # 使用回测引擎提供的收盘价数据
            if 'open_prices' in self.context and security in self.context['open_prices']:
                return self.context['open_prices'][security]

            data = get_price(security, count=1, fields=['close'], end_date=date)
            return data['close'].iloc[-1] if len(data) > 0 else None
        except Exception as e:
            log.error(f"获取 {security} 价格失败: {e}")
            return None

    def _get_open_price(self, security, date):
        """获取股票开盘价"""
        try:
            # 使用回测引擎提供的开盘价数据
            if 'open_prices' in self.context and security in self.context['open_prices']:
                return self.context['open_prices'][security]

            data = get_price(security, count=1, fields=['open'], end_date=date)
            return data['open'].iloc[-1] if len(data) > 0 else None
        except Exception as e:
            log.error(f"获取 {security} 开盘价失败: {e}")
            return None

    def _print_account_status(self, date):
        """打印账户状态"""
        account = self.context['account']
        cash = account.cash
        position_value = sum(
            self._get_current_price(sec, date) * amt
            for sec, amt in account.positions.items()
            if self._get_current_price(sec, date)
        )
        total_assets = cash + position_value
        log.info(f"[{date}] 现金: {cash:.2f}, 持仓市值: {position_value:.2f}, 总资产: {total_assets:.2f}")

    def calculate_buy_amount(self, target_value, price):
        """计算可买入数量（不考虑手续费）"""
        if price <= 0 or target_value <= 0:
            return 0
        amount = int(target_value / price)
        return amount if amount > 0 else 0

    def print_learning_summary(self):
        """打印学习总结"""
        if hasattr(self.agent, 'print_learning_status'):
            self.agent.print_learning_status()
        else:
            log.warning("Agent没有学习状态打印方法")
# 修改后的Strategy_Core.py
from Utilities import log
import pandas as pd
import numpy as np
from Data_Handling import get_weight, get_price
from Agent import Agent  # 导入Agent类


class WeightBasedStrategy:
    def __init__(self, context):
        self.context = context
        self.g = type('Global', (object,), {})  # 模拟全局变量
        self.g.securities = []  # 中证500成分股
        self.g.weights = {}  # 股票权重
        self.g.is_initial_purchase_done = False  # 初始半仓标记
        self.g.initial_half_pos = {}  # 初始半仓持仓 {股票代码: 数量}

        # 初始化Agent（参数可调整）
        self.agent = Agent(account=context.account,
                           data_handler=context.data_handler,
                           Epsilon=0.1, Alpha=0.5)
        self.g.current_state = [0, 0, 0]  # Agent的初始状态（可根据实际因子调整）

    def initialize(self):
        """初始化策略，先建立半仓底仓"""
        log.info('策略初始化：建立半仓底仓')
        try:
            weight_df = get_weight()
            self.g.securities = weight_df['ts_code'].unique().tolist()
            self.g.weights = dict(zip(weight_df['ts_code'], weight_df['weight']))
            log.info(f"股票池包含 {len(self.g.securities)} 只中证500成分股")
        except Exception as e:
            log.error(f"初始化失败：{str(e)}")

    def before_market_open(self, date):
        """开盘前决策：根据Agent输出确定当日策略"""
        # 获取当前状态（示例：使用前3日指数涨跌幅作为状态，需根据实际因子实现）
        self.g.current_state = self._get_market_state(date)
        # Agent决策（1=看多做T，0=看空做T）
        self.g.agent_decision = self.agent.decide(self.g.current_state)
        log.info(f"Agent决策：{self.g.agent_decision}（1=看多做T，0=看空做T）")

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

        # 打印当日账户状态
        self._print_account_status(date)

    def _initial_half_position(self, date):
        """建立初始半仓（50%仓位）"""
        account = self.context['account']
        total_cash = account.cash * 0.5  # 仅用一半资金
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
                log.info(f"初始半仓买入 {security}：{buy_amount}股")

    def _open_buy_half(self, date):
        """开盘买入半仓（基于初始半仓的同等金额）"""
        account = self.context['account']
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
                log.info(f"看多策略买入 {security}：{buy_amount}股")

    def _open_sell_half(self, date):
        """开盘卖出全部初始半仓"""
        account = self.context['account']
        for security, amount in self.g.initial_half_pos.items():
            if security in account.positions and account.positions[security] >= amount:
                current_price = self._get_current_price(security, date)
                if not current_price:
                    continue
                if account.sell(date, security, current_price, amount):
                    log.info(f"看空策略卖出 {security}：{amount}股")

    def _close_sell_half_if_no_profit(self, date):
        """收盘时，若未盈利1%则卖出当日新增仓位"""
        account = self.context['account']
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
                    log.info(f"收盘卖出 {security}：{sell_amount}股（未达盈利1%）")

    def _close_buy_half_if_no_drop(self, date):
        """收盘时，若指数未跌1%则买入半仓"""
        if self._get_index_drop(date) < 0.01:  # 指数跌幅未达1%
            self._open_buy_half(date)  # 复用买入逻辑

    def _get_current_price(self, security, date):
        """获取股票当前价格（收盘价）"""
        try:
            data = get_price(security, count=1, fields=['close'], end_date=date)
            return data['close'].iloc[-1] if len(data) > 0 else None
        except Exception as e:
            log.error(f"获取 {security} 价格失败：{e}")
            return None

    def _get_open_price(self, security, date):
        """获取股票开盘价"""
        try:
            data = get_price(security, count=1, fields=['open'], end_date=date)
            return data['open'].iloc[-1] if len(data) > 0 else None
        except Exception as e:
            log.error(f"获取 {security} 开盘价失败：{e}")
            return None

    def _get_index_drop(self, date):
        """获取当日指数跌幅（示例：中证500）"""
        try:
            # 需实现指数数据获取逻辑
            index_data = get_price('000905.SH', count=1, fields=['open', 'close'], end_date=date)
            if len(index_data) == 0:
                return 0
            drop_ratio = (index_data['open'].iloc[-1] - index_data['close'].iloc[-1]) / index_data['open'].iloc[-1]
            return drop_ratio
        except Exception as e:
            log.error(f"获取指数跌幅失败：{e}")
            return 0

    def _get_market_state(self, date):
        """获取市场状态（供Agent决策使用）"""
        # 示例：返回前3日指数涨跌幅状态（需根据实际因子实现）
        return [0, 0, 0]  # 实际应用中需替换为真实状态计算

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
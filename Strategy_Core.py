# file:D:\read\task\回测框架搭建\Strategy_Core.py
from Utilities import log
import pandas as pd
from Data_Handling import get_weight  # 导入获取权重的函数


# 导入获取权重的函数
class WeightBasedStrategy:
    def __init__(self, context):
        self.context = context
        self.g = type('Global', (object,), {})  # 模拟全局变量g
        self.g.securities = []  # 存储中证500成分股
        self.g.weights = {}  # 存储股票权重 {股票代码: 权重}
        self.g.is_initial_purchase_done = False  # 标记是否完成初始购买

    def initialize(self):
        """初始化策略"""
        log.info('初始函数开始运行且全局只运行一次')

        # 获取中证500成分股及权重（默认使用最新日期数据）
        try:
            weight_df = get_weight()
            # 去重并提取股票代码和权重
            self.g.securities = weight_df['ts_code'].unique().tolist()
            self.g.weights = dict(zip(weight_df['ts_code'], weight_df['weight']))

            log.info(f"策略初始化完成，股票池包含 {len(self.g.securities)} 只中证500成分股")
            log.info("策略规则：半仓按权重购买所有成分股后长期持有，不再进行买卖操作")
        except Exception as e:
            log.error(f"初始化失败：{str(e)}")

    def before_market_open(self, date):
        """开盘前运行"""
        pass

    def market_open(self, date):
        """开盘时运行"""
        log.info(f'函数运行时间(market_open)：{str(date)}')

        # 只在第一个交易日执行初始购买
        if not self.g.is_initial_purchase_done:
            self._initial_purchase(date)
            self.g.is_initial_purchase_done = True  # 标记为已完成
        else:
            log.info("已完成初始购买，今日无交易操作")

    def _initial_purchase(self, date):
        """半仓按权重购买所有成分股"""
        if not self.g.securities or not self.g.weights:
            log.error("股票池或权重数据为空，无法执行初始购买")
            return

        account = self.context['account']
        total_cash = account.cash
        total_weight = sum(self.g.weights.values())  # 总权重（用于归一化）

        if total_weight <= 0:
            log.error("权重总和无效，无法计算购买比例")
            return

        # 使用半仓资金进行购买
        used_cash = total_cash * 0.5
        log.info(f"开始执行半仓购买，总资金：{total_cash:.2f}，使用资金：{used_cash:.2f}")

        # 遍历所有成分股按权重购买
        for security in self.g.securities:
            weight = self.g.weights.get(security, 0)
            if weight <= 0:
                log.warning(f"股票 {security} 权重为0，跳过购买")
                continue

            # 计算该股票的配置金额（按权重比例分配半仓资金）
            allocation_ratio = weight / total_weight
            target_value = used_cash * allocation_ratio
            """
            log.info(f"股票 {security} 权重：{weight}，配置金额：{target_value:.2f}")
            """
            # 获取当前价格
            from Data_Handling import get_price
            current_data = get_price(security, count=1, fields=['close'], end_date=date)
            if len(current_data) == 0:
                log.error(f"无法获取 {security} 价格数据，跳过购买")
                continue

            current_price = current_data['close'].iloc[-1]
            if current_price <= 0:
                log.error(f"股票 {security} 价格无效（{current_price}），跳过购买")
                continue

            # 计算可购买数量（考虑手续费）
            buy_amount = self.calculate_buy_amount(target_value, current_price)
            if buy_amount <= 0:
                log.warning(f"股票 {security} 可购买数量为0，跳过购买")
                continue

            # 执行购买
            success = account.buy(date, security, current_price, buy_amount)
            """
            if success:
                log.info(
                    f"✅ 买入 {security}，价格：{current_price:.2f}，数量：{buy_amount}，花费：{current_price * buy_amount:.2f}")
            """
            if not success:
                log.error(f"❌ 买入 {security} 失败")

        # 更新上下文现金信息
        self.context['portfolio']['available_cash'] = account.cash
        log.info(f"半仓购买完成，剩余现金：{account.cash:.2f}")

    def calculate_buy_amount(self, target_value, price):
        """根据目标金额计算可买入数量（考虑手续费）"""
        if price <= 0 or target_value <= 0:
            return 0

        # 估算最大可买数量（不考虑手续费）
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

    def after_market_close(self, date):
        """收盘后运行"""
        log.info(f'函数运行时间(after_market_close)：{str(date)}')

        # 打印账户状态
        account = self.context['account']
        cash = account.cash
        total_assets = cash

        # 计算持仓市值
        position_value = 0
        for security, amount in account.positions.items():
            from Data_Handling import get_price
            current_data = get_price(security, count=1, fields=['close'], end_date=date)
            if len(current_data) > 0:
                current_price = current_data['close'].iloc[-1]
                value = current_price * amount
                position_value += value
                """
                log.info(
                    f"持仓情况: {security} - 数量: {amount}, 当前价格: {current_price:.2f}, 持仓市值: {value:.2f}")
                """
        total_assets = cash + position_value

        log.info(f"账户状态 - 现金: {cash:.2f}, 持仓市值: {position_value:.2f}, 总资产: {total_assets:.2f}")
        log.info('一天结束\n')
"""
        # 打印当日交易记录
        if account.trade_history:
            today_trades = [trade for trade in account.trade_history
                            if pd.to_datetime(trade['date']).date() == date.date()]
            for trade in today_trades:
                log.info(f'当日成交记录：{trade}')
        else:
            log.info('当日无成交记录')
"""
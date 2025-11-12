# file:D:\read\task\回测框架搭建\Strategy_Core.py
from Utilities import log
import pandas as pd


class MA5Strategy:
    def __init__(self, context):
        self.context = context
        self.g = type('Global', (object,), {})  # 模拟全局变量g
        self.g.securities = ['600612.SH']  # 仅包含一只股票的股票池
        self.g.previous_prices = {}  # 存储每只股票前一天的价格

    def initialize(self):
        """初始化策略"""
        log.info('初始函数开始运行且全局只运行一次')

        log.info(f"策略初始化完成，股票池包含 {len(self.g.securities)} 只股票")
        log.info("策略规则：仅交易600612.SH一只股票")

    def before_market_open(self, date):
        """开盘前运行"""
        pass

    def market_open(self, date):
        """开盘时运行"""
        log.info(f'函数运行时间(market_open)：{str(date)}')

        # 记录当天买入的股票数量
        bought_stocks_count = 0
        max_bought_stocks = 1  # 每日最多买入1只股票（因为我们只有一只）

        # 对股票池中的每只股票执行交易逻辑
        for security in self.g.securities:
            if bought_stocks_count >= max_bought_stocks:
                log.info(f"已达到每日最大买入限制 ({max_bought_stocks}只)，停止买入")
                break

            # 调用DataHandler的get_price获取当前价格
            from Data_Handling import get_price
            current_data = get_price(security, count=1, fields=['close'], end_date=date)

            if len(current_data) == 0:
                continue  # 如果无法获取当前价格数据，则跳过这只股票

            # 获取当前价格
            current_price = current_data['close'].iloc[-1]
            cash = self.context['portfolio']['available_cash']
            account = self.context['account']

            # 获取前一天价格
            previous_price = self.g.previous_prices.get(security, None)

            # 如果有前一天的价格数据，执行交易逻辑
            if previous_price is not None:
                # 今日股价比昨日高则买入
                if current_price > previous_price:
                    # 检查是否已经达到最大持股数量限制
                    if not self.check_holding_limit(account):
                        log.info(f"已达到最大持股数量限制，跳过买入 {security}")
                        continue

                    # 调用交易函数执行买入
                    success = self.trading_function(
                        date=date,
                        security=security,
                        action='buy',
                        price=current_price,
                        cash=cash,
                        account=account
                    )

                    if success:
                        bought_stocks_count += 1
                # 否则卖出（今日股价不高于昨日）
                elif security in account.positions and account.positions[security] > 0:
                    # 调用交易函数执行卖出
                    self.trading_function(
                        date=date,
                        security=security,
                        action='sell',
                        price=current_price,
                        cash=cash,
                        account=account
                    )

            # 更新前一天价格为今天的价格（供明天使用）
            self.g.previous_prices[security] = current_price

    def trading_function(self, date, security, action, price, cash, account):
        """统一处理买入卖出的交易函数"""
        if action == 'buy':
            if cash > 0:
                # 计算可买数量（考虑手续费）
                buy_amount = self.calculate_buy_amount(cash, price)
                if buy_amount > 0:
                    success = account.buy(date, security, price, buy_amount)
                    if success:
                        log.info(f"🎯 买入信号触发！买入 {security}，价格：{price:.2f}，数量：{buy_amount}")
                        # 更新现金信息
                        self.context['portfolio']['available_cash'] = account.cash
                        return True
                    else:
                        log.info(f"买入失败，可能由于现金不足")
                else:
                    log.info(f"计算出的买入数量为0，跳过买入")
            else:
                log.info(f"今日价格高于昨日，但现金不足，无法买入")
            return False

        elif action == 'sell':
            # 检查是否有持仓
            has_position = security in account.positions and account.positions[security] > 0
            log.info(
                f"检查持仓: {security} 在持仓中: {security in account.positions}, 持仓数量: {account.positions.get(security, 0)}")

            if has_position:
                sell_amount = account.positions[security]  # 卖出全部持仓
                success = account.sell(date, security, price, sell_amount)
                if success:
                    log.info(f"📉 卖出信号触发！卖出 {security}，价格：{price:.2f}，数量：{sell_amount}")
                    return True
                else:
                    log.info(f"卖出失败")
            else:
                log.info(f"今日价格不高于昨日，但无持仓可卖，跳过交易")
            return False

    def calculate_buy_amount(self, cash, price):
        """计算可买入数量（考虑手续费）"""
        # 估算手续费（买入佣金万分之三，最低5元）
        # 先计算不考虑手续法的最大数量
        max_amount = int(cash / price)

        # 如果最大数量为0，直接返回0
        if max_amount == 0:
            return 0

        # 计算手续费
        cost = price * max_amount
        commission = max(0.0003 * cost, 5)
        total_cost = cost + commission

        # 如果总成本超过现金，减少买入数量
        while total_cost > cash and max_amount > 0:
            max_amount -= 1
            cost = price * max_amount
            commission = max(0.0003 * cost, 5)
            total_cost = cost + commission

        return max_amount

    def check_holding_limit(self, account):
        """检查是否达到最大持股数量限制"""
        max_stock_holdings = self.context['portfolio'].get('max_stock_holdings')
        if max_stock_holdings is None:
            return True  # 无限制时返回True表示可以买入
        # 当前持股数量小于等于最大限制时返回True
        return len(account.positions) < max_stock_holdings

    def after_market_close(self, date):
        """收盘后运行"""
        log.info(f'函数运行时间(after_market_close)：{str(date)}')

        # 打印账户状态
        account = self.context['account']
        cash = account.cash
        total_assets = cash

        # 计算持仓市值（这里简化处理，只显示部分持仓）
        position_value = 0
        for security, amount in account.positions.items():
            from Data_Handling import get_price
            current_data = get_price(security, count=1, fields=['close'], end_date=date)
            if len(current_data) > 0:
                current_price = current_data['close'].iloc[-1]
                value = current_price * amount
                position_value += value
                log.info(
                    f"持仓情况: {security} - 数量: {amount}, 当前价格: {current_price:.2f}, 持仓市值: {value:.2f}")

        total_assets = cash + position_value

        log.info(f"账户状态 - 现金: {cash:.2f}, 持仓市值: {position_value:.2f}, 总资产: {total_assets:.2f}")

        # 打印交易历史
        if account.trade_history:
            # 只打印当天的交易记录
            today_trades = [trade for trade in account.trade_history
                            if pd.to_datetime(trade['date']).date() == date.date()]
            for trade in today_trades:
                log.info(f'当日成交记录：{trade}')
        else:
            log.info('当日无成交记录')

        log.info('一天结束\n')

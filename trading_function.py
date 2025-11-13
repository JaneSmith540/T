import pandas as pd
import numpy as np
from Utilities import log


class Order:
    """订单类，用于记录订单信息"""
    ORDER_ID = 0  # 类变量，用于生成唯一订单ID

    def __init__(self, security, amount, style=None, side='long', pindex=0, close_today=False):
        Order.ORDER_ID += 1
        self.order_id = Order.ORDER_ID
        self.security = security  # 股票代码
        self.amount = amount  # 数量（正数为买入，负数为卖出）
        self.original_amount = amount  # 原始委托数量
        self.style = style  # 下单方式（如市价、限价等）
        self.side = side  # 多空方向
        self.pindex = pindex
        self.close_today = close_today
        self.status = 'open'  # 订单状态：open, filled, cancelled, partial
        self.filled_amount = 0  # 已成交数量
        self.filled_price = 0.0  # 成交均价
        self.create_time = pd.Timestamp.now()  # 创建时间
        self.fill_time = None  # 成交时间

    def __repr__(self):
        return (f"Order(id={self.order_id}, security={self.security}, amount={self.amount}, "
                f"status={self.status}, filled={self.filled_amount})")


class TradingFunctions:
    def __init__(self, context):
        self.context = context
        self.orders = []  # 所有订单列表
        self.trades = []  # 所有成交记录

    def order(self, security, amount, style=None, side='long', pindex=0, close_today=False):
        """
        按股数下单
        :param security: 股票代码
        :param amount: 下单数量（正数为买入，负数为卖出）
        :param style: 下单方式
        :param side: 多空方向
        :param pindex: 价格指数
        :param close_today: 是否平今
        :return: 订单对象
        """
        if amount == 0:
            log.warning("下单数量不能为0")
            return None

        # 创建订单
        order = Order(
            security=security,
            amount=amount,
            style=style,
            side=side,
            pindex=pindex,
            close_today=close_today
        )

        # 尝试立即成交（简化处理，实际回测中可能需要根据市场情况处理）
        self._execute_order(order)

        self.orders.append(order)
        log.info(f"创建订单: {order}")
        return order

    def order_target(self, security, amount, style=None, side='long', pindex=0, close_today=False):
        """
        目标股数下单（计算与当前持仓的差额并下单）
        :param security: 股票代码
        :param amount: 目标持仓数量
        :param style: 下单方式
        :param side: 多空方向
        :param pindex: 价格指数
        :param close_today: 是否平今
        :return: 订单对象
        """
        account = self.context['account']
        current_amount = account.positions.get(security, 0)
        order_amount = amount - current_amount
        if order_amount == 0:
            log.info(f"目标持仓已达，无需下单: {security}")
            return None
        return self.order(security, order_amount, style, side, pindex, close_today)

    def order_value(self, security, value, style=None, side='long', pindex=0, close_today=False):
        """
        按价值下单
        :param security: 股票代码
        :param value: 下单金额（正数为买入，负数为卖出）
        :param style: 下单方式
        :param side: 多空方向
        :param pindex: 价格指数
        :param close_today: 是否平今
        :return: 订单对象
        """
        if value == 0:
            log.warning("下单金额不能为0")
            return None

        # 获取当前价格
        from Data_Handling import get_price
        current_data = get_price(security, count=1, fields=['Clsprc'], end_date=self.context['current_dt'])
        if len(current_data) == 0:
            log.error(f"无法获取 {security} 价格数据，下单失败")
            return None

        current_price = current_data['Clsprc'].iloc[-1]
        if current_price <= 0:
            log.error(f"无效价格: {current_price}，下单失败")
            return None

        # 计算股数（向下取整，确保金额不超过指定值）
        amount = int(abs(value) / current_price)
        if amount <= 0:
            log.warning(f"计算出的下单数量为0: {security}")
            return None

        # 保持与价值相同的方向（正为买，负为卖）
        amount = amount if value > 0 else -amount
        return self.order(security, amount, style, side, pindex, close_today)

    def order_target_value(self, security, value, style=None, side='long', pindex=0, close_today=False):
        """
        目标价值下单
        :param security: 股票代码
        :param value: 目标持仓价值
        :param style: 下单方式
        :param side: 多空方向
        :param pindex: 价格指数
        :param close_today: 是否平今
        :return: 订单对象
        """
        # 获取当前价格
        from Data_Handling import get_price
        current_data = get_price(security, count=1, fields=['Clsprc'], end_date=self.context['current_dt'])
        if len(current_data) == 0:
            log.error(f"无法获取 {security} 价格数据，下单失败")
            return None

        current_price = current_data['Clsprc'].iloc[-1]
        if current_price <= 0:
            log.error(f"无效价格: {current_price}，下单失败")
            return None

        # 计算目标股数
        target_amount = int(abs(value) / current_price) if current_price != 0 else 0
        return self.order_target(security, target_amount, style, side, pindex, close_today)

    def cancel_order(self, order):
        """
        撤单
        :param order: 订单对象
        :return: 是否成功
        """
        if order not in self.orders:
            log.error("订单不存在")
            return False

        if order.status != 'open' and order.status != 'partial':
            log.warning(f"订单 {order.order_id} 无法撤销，当前状态: {order.status}")
            return False

        order.status = 'cancelled'
        log.info(f"订单已撤销: {order}")
        return True

    def get_open_orders(self):
        """
        获取未完成订单
        :return: 未完成订单列表
        """
        return [order for order in self.orders if order.status in ['open', 'partial']]

    def get_orders(self, order_id=None, security=None, status=None):
        """
        获取订单信息
        :param order_id: 订单ID
        :param security: 股票代码
        :param status: 订单状态
        :return: 符合条件的订单列表
        """
        result = self.orders

        if order_id:
            result = [order for order in result if order.order_id == order_id]
        if security:
            result = [order for order in result if order.security == security]
        if status:
            result = [order for order in result if order.status == status]

        return result

    def get_trades(self):
        """
        获取成交信息
        :return: 成交记录列表
        """
        return self.trades.copy()

    def _execute_order(self, order):
        """
        执行订单（内部方法）
        :param order: 订单对象
        """
        account = self.context['account']
        date = self.context['current_dt']

        # 获取当前价格
        from Data_Handling import get_price
        current_data = get_price(order.security, count=1, fields=['Clsprc'], end_date=date)
        if len(current_data) == 0:
            order.status = 'failed'
            log.error(f"订单执行失败，无法获取 {order.security} 价格数据")
            return

        current_price = current_data['Clsprc'].iloc[-1]

        # 处理买入订单
        if order.amount > 0:
            # 计算可买入数量（考虑手续费）
            max_possible = self._calculate_max_buy_amount(account.cash, current_price)
            if max_possible <= 0:
                order.status = 'failed'
                log.error("买入失败，现金不足或计算出错")
                return

            # 实际成交数量（不超过委托数量）
            fill_amount = min(order.amount, max_possible)

            # 执行买入
            success = account.buy(date, order.security, current_price, fill_amount)
            if success:
                self._record_trade(order, fill_amount, current_price, date)
                order.filled_amount = fill_amount
                order.filled_price = current_price
                order.fill_time = date
                order.status = 'filled' if fill_amount == order.amount else 'partial'

        # 处理卖出订单
        elif order.amount < 0:
            # 需要卖出的数量（取绝对值）
            sell_amount = abs(order.amount)
            current_holdings = account.positions.get(order.security, 0)

            if current_holdings < sell_amount:
                fill_amount = current_holdings
                if fill_amount <= 0:
                    order.status = 'failed'
                    log.error("卖出失败，无持仓")
                    return
            else:
                fill_amount = sell_amount

            # 执行卖出
            success = account.sell(date, order.security, current_price, fill_amount)
            if success:
                self._record_trade(order, -fill_amount, current_price, date)  # 用负数表示卖出
                order.filled_amount = fill_amount
                order.filled_price = current_price
                order.fill_time = date
                order.status = 'filled' if fill_amount == sell_amount else 'partial'

    def _calculate_max_buy_amount(self, cash, price):
        """计算最大可买入数量（考虑手续费）"""
        if price <= 0 or cash <= 0:
            return 0
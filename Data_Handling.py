# file:D:\read\task\Data_Handling.py
import pandas as pd
import os
import numpy as np
from datetime import datetime

# 全局数据处理器实例，避免重复加载
_data_handler_instance = None


def get_data_handler(file_path=None, index_file_path=None):
    """获取全局数据处理器实例"""
    global _data_handler_instance
    if _data_handler_instance is None and file_path:
        _data_handler_instance = DataHandler(file_path, index_file_path)
    return _data_handler_instance


# 对外提供的查询接口，内部使用全局数据处理器
def get_price(security, start_date=None, end_date=None, fields=None, count=None):
    dh = get_data_handler()
    if dh:
        return dh.get_price(security, start_date, end_date, fields, count)
    raise RuntimeError("数据处理器未初始化，请先创建DataHandler实例")


def get_index_price(start_date=None, end_date=None, fields=None, count=None):
    """获取中证500指数价格数据的对外接口"""
    dh = get_data_handler()
    if dh:
        return dh.get_index_price(start_date, end_date, fields)
    raise RuntimeError("数据处理器未初始化，请先创建DataHandler实例")


def get_weight():
    dh = get_data_handler()
    if dh:
        return dh.get_weight()
    raise RuntimeError("数据处理器未初始化，请先创建DataHandler实例")


def get_all_securities(date=None):
    dh = get_data_handler()
    if dh and dh.weights_data:
        return list(dh.weights_data.keys())
    return []


class StockData:
    def __init__(self):
        self.Stkcd = None  # 股票代码
        self.Opnprc = None  # 开盘价
        self.Hiprc = None  # 最高价
        self.Loprc = None  # 最低价
        self.Clsprc = None  # 收盘价
        self.Trdsta = None  # 交易状态
        self.LimitDown = None  # 跌停价
        self.LimitUp = None  # 涨停价
        self.Dnshrtrd = None  # 交易量
        self.Dsmvosd = None  # 流通市值


class DataHandler:
    def __init__(self, file_path, index_file_path=None):
        self.file_path = file_path
        self.index_file_path = index_file_path or r"C:\Users\chanpi\Desktop\task\中证500指数_201801-202506.csv"
        self.all_stock_data = None  # 预加载的所有股票数据
        self.weights_data = None  # 预加载的权重数据
        self.dates = None  # 所有交易日
        self.index_data = None  # 预加载的指数数据
        self._preload_data()  # 初始化时预加载所有数据

    def _preload_data(self):
        """预加载所有股票数据和权重数据到内存"""
        # 加载股票价格数据
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"数据文件不存在: {self.file_path}")

        # 读取并预处理股票数据
        df = pd.read_pickle(self.file_path)

        # 处理日期列
        date_column = 'trade_date'
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df = df.dropna(subset=[date_column])

        # 处理股票代码（统一格式为xxx.SH/xxx.SZ）
        code_column = 'ts_code'
        df[code_column] = df[code_column].astype(str)

        # 确保价格字段为数值类型
        price_fields = ['open', 'high', 'low', 'close']
        for field in price_fields:
            df[field] = pd.to_numeric(df[field], errors='coerce')

        # 设置复合索引（日期+股票代码），加速查询
        self.all_stock_data = df.set_index(['trade_date', 'ts_code']).sort_index()

        # 提取所有交易日
        self.dates = pd.DatetimeIndex(self.all_stock_data.index.unique(level=0)).sort_values()

        # 预加载权重数据
        self._preload_weights()

        # 预加载指数数据
        self._preload_index_data()

    def _preload_weights(self):
        """预加载中证500成分股权重数据"""
        weight_file_path = r"D:\read\task\中证500成分股,单一股票数据.csv"
        if os.path.exists(weight_file_path):
            try:
                df = pd.read_csv(weight_file_path, dtype=str)
                # 假设权重列名为'weight'，股票代码列名为'con_code'
                df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
                df = df.dropna(subset=['con_code', 'weight'])
                # 转换股票代码格式与价格数据一致
                df['con_code'] = df['con_code'].astype(str)
                self.weights_data = df.set_index('con_code')['weight'].to_dict()
            except Exception as e:
                print(f"权重数据加载警告: {str(e)}")
                self.weights_data = {}
        else:
            self.weights_data = {}

    def _preload_index_data(self):
        """预加载指数数据到内存"""
        if not os.path.exists(self.index_file_path):
            print(f"指数数据文件不存在: {self.index_file_path}")
            self.index_data = None
            return

        try:
            # 读取CSV文件
            df = pd.read_csv(self.index_file_path)
            print(f"原始指数数据列: {df.columns.tolist()}")

            # 处理日期列 - 根据实际数据格式
            if 'trade_date' in df.columns:
                # 将YYYYMMDD格式的日期转换为datetime
                df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
                df = df.set_index('trade_date').sort_index()
                print(f"预加载指数数据成功，日期范围: {df.index.min()} 至 {df.index.max()}")
            else:
                print("警告: 指数数据文件中没有找到'trade_date'列")
                print(f"可用列: {df.columns.tolist()}")
                self.index_data = None
                return

            self.index_data = df
            print(f"预加载指数数据成功，共 {len(df)} 条记录")
            print(f"指数数据列: {df.columns.tolist()}")

            # 检查是否有2022-09-08的数据
            target_date = pd.to_datetime('2022-09-08')
            if target_date in self.index_data.index:
                print(f"找到目标日期 {target_date} 的指数数据")
            else:
                print(f"警告: 未找到目标日期 {target_date} 的指数数据")
                print(
                    f"最接近的日期: {self.index_data.index[self.index_data.index <= target_date][-1] if len(self.index_data.index[self.index_data.index <= target_date]) > 0 else '无'}")

        except Exception as e:
            print(f"指数数据加载错误: {str(e)}")
            import traceback
            traceback.print_exc()
            self.index_data = None

    def get_previous_trading_day(self, current_date):
        """获取当前日期的上一个有效交易日"""
        current_date = pd.to_datetime(current_date)
        previous_days = self.dates[self.dates < current_date]
        return previous_days[-1] if len(previous_days) > 0 else None

    def get_stock_data(self):
        """获取所有股票数据（用于提取日期列表）"""
        return self.all_stock_data.reset_index().set_index('trade_date')

    def get_single_day_data(self, date):
        """获取某一天所有股票的收盘价"""
        date = pd.to_datetime(date)
        try:
            return self.all_stock_data.loc[date]['close']
        except KeyError:
            return pd.Series([np.nan], index=[None])

    def get_single_day_open_data(self, date):
        """获取某一天所有股票的开盘价"""
        date = pd.to_datetime(date)
        try:
            return self.all_stock_data.loc[date]['open']
        except KeyError:
            return pd.Series([np.nan], index=[None])

    def get_price(self, security, start_date=None, end_date=None, fields=None, count=None):
        """
        从内存中查询股票价格数据
        :param security: 股票代码
        :param start_date: 开始日期
        :param end_date: 结束日期
        :param fields: 需要的字段列表
        :param count: 返回的记录数量
        :return: 价格数据DataFrame
        """
        security = str(security)
        start_date = pd.to_datetime(start_date) if start_date else None
        end_date = pd.to_datetime(end_date) if end_date else None

        # 基础查询条件：股票代码匹配
        mask = self.all_stock_data.index.get_level_values('ts_code') == security
        filtered = self.all_stock_data[mask]

        # 日期过滤
        if start_date:
            filtered = filtered[filtered.index.get_level_values('trade_date') >= start_date]
        if end_date:
            filtered = filtered[filtered.index.get_level_values('trade_date') <= end_date]

        # 字段过滤
        available_fields = ['open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount']
        if fields:
            valid_fields = [f for f in fields if f in available_fields]
            filtered = filtered[valid_fields]

        # 限制返回数量
        if count and count > 0:
            filtered = filtered.tail(count)

        # 重置索引为日期，便于策略使用
        return filtered.reset_index(level='ts_code', drop=True)

    def get_weight(self):
        """获取预加载的权重数据"""
        if not self.weights_data:
            self._preload_weights()
        # 转换为DataFrame格式返回
        return pd.DataFrame(list(self.weights_data.items()), columns=['ts_code', 'weight'])

    def get_index_price(self, start_date, end_date, fields):
        """获取中证500指数价格数据"""
        # 读取CSV文件
        df = pd.read_csv(r"C:\Users\chanpi\Desktop\task\中证500指数_201601-202506.csv")

        # 关键修复1：使用实际日期列名'trade_date'
        if 'trade_date' in df.columns:
            # 关键修复2：显式指定格式为'YYYYMMDD'，确保解析正确
            df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d', errors='coerce')
            # 移除解析失败的无效日期
            df = df.dropna(subset=['trade_date'])

            # 按日期筛选（使用正确的列名）
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            mask = (df['trade_date'] >= start) & (df['trade_date'] <= end)
            df = df.loc[mask]

        # 字段筛选（保持不变）
        if fields and len(fields) > 0:
            available_fields = [field for field in fields if field in df.columns]
            if available_fields:
                df = df[available_fields]

        # 若筛选后无数据，打印警告（便于调试）
        if df.empty:
            import logging
            logging.warning(f"[{start_date}] 无法获取指数数据（日期范围或格式错误）")

        return df
    def get_index_data_for_date(self, date):
        """
        专门为单个日期获取指数数据的方法
        返回包含开盘、最高、最低、收盘价的字典
        """
        try:
            date = pd.to_datetime(date)

            if self.index_data is None:
                print(f"警告: 指数数据未加载，无法获取 {date} 的数据")
                return {'open': 0, 'high': 0, 'low': 0, 'close': 0}

            # 直接使用索引获取单日数据
            if date in self.index_data.index:
                row = self.index_data.loc[date]

                # 构建返回字典，处理可能的列名差异
                result = {}

                # 尝试不同的列名
                open_col = next((col for col in ['open', 'Open', 'OPEN'] if col in row.index), None)
                high_col = next((col for col in ['high', 'High', 'HIGH'] if col in row.index), None)
                low_col = next((col for col in ['low', 'Low', 'LOW'] if col in row.index), None)
                close_col = next((col for col in ['close', 'Close', 'CLOSE'] if col in row.index), None)

                result['open'] = float(row[open_col]) if open_col else 0
                result['high'] = float(row[high_col]) if high_col else 0
                result['low'] = float(row[low_col]) if low_col else 0
                result['close'] = float(row[close_col]) if close_col else 0

                return result
            else:
                print(f"警告: 未找到日期 {date} 的指数数据")
                # 尝试找到最接近的日期
                previous_dates = self.index_data.index[self.index_data.index <= date]
                if len(previous_dates) > 0:
                    closest_date = previous_dates[-1]
                    print(f"使用最接近的日期: {closest_date}")
                    return self.get_index_data_for_date(closest_date)
                else:
                    return {'open': 0, 'high': 0, 'low': 0, 'close': 0}

        except Exception as e:
            print(f"获取指定日期指数数据错误: {e}")
            import traceback
            traceback.print_exc()
            return {'open': 0, 'high': 0, 'low': 0, 'close': 0}

    def get_index_close_price(self, date):
        """
        获取指定日期的指数收盘价
        """
        try:
            index_data = self.get_index_data_for_date(date)
            return index_data.get('close', 0)
        except Exception as e:
            print(f"获取指数收盘价错误: {e}")
            return 0
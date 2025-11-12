import pandas as pd
import os
from datetime import datetime
import numpy as np



class StockData:
    def __init__(self):
        self.Stkcd = None  # 股票代码
        self.Opnprc = None  # 开盘价
        self.Hiprc = None  # 最高价
        self.Loprc = None  # 最低价
        self.Clsprc = None  # 收盘价
        self.Trdsta = None  # 交易状态
        # 1 = 正常交易，2 = ST，3＝*ST，4＝S（2006年10月9日及之后股改未完成），5＝SST，6＝S * ST，7 = G（2006年10月9日之前已完成股改），8 = GST，9 = G * ST，10 = U（2006年10月9日之前股改未完成），11 = UST，12 = U * ST，13 = N，14 = NST，15 = N * ST，16 = PT
        self.LimitDown = None  # 跌停价
        self.LimitUp = None  # 涨停价
        self.Dnshrtrd = None  # 交易量
        self.Dsmvosd = None  # 流通市值
"""
下面的很合我意了
"""

def get_price(security, start_date=None, end_date=None, frequency='daily', fields=None,
              skip_paused=False, count=None, panel=True, fill_paused=True):
    """
    获取历史数据，可查询多个标的多个数据字段，返回数据格式为 DataFrame
    """
    # 数据文件路径
    file_path = r"D:\read\task\机器学习数据.pkl"

    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    # 读取pkl文件
    df = pd.read_pickle(file_path)

    # 转换数值字段为适当类型
    numeric_fields = ['open', 'high', 'low', 'close', 'pre_close',
                      'change', 'pct_chg', 'vol', 'amount']
    for field in numeric_fields:
        if field in df.columns:
            try:
                df[field] = pd.to_numeric(df[field])
            except (ValueError, TypeError):
                pass

    # 转换日期列的格式
    if 'trade_date' in df.columns:
        # 假设日期是整数格式如20180102，转换为datetime
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')

    # 处理股票代码
    if 'ts_code' in df.columns:
        # 确保股票代码是字符串格式
        df['ts_code'] = df['ts_code'].astype(str)

    # 过滤股票代码
    if isinstance(security, list):
        # 多个股票代码
        securities = [str(s) for s in security]
        df = df[df['ts_code'].isin(securities)]
    else:
        # 单个股票代码
        security_str = str(security)
        df = df[df['ts_code'] == security_str]

    # 过滤日期范围
    if start_date:
        start_date = pd.to_datetime(start_date)
        df = df[df['trade_date'] >= start_date]

    if end_date:
        end_date = pd.to_datetime(end_date)
        df = df[df['trade_date'] <= end_date]

    # 选择需要的字段
    available_fields = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close',
                        'pre_close', 'change', 'pct_chg', 'vol', 'amount']

    if fields:
        # 检查字段是否有效
        invalid_fields = [f for f in fields if f not in available_fields]
        if invalid_fields:
            raise ValueError(f"无效的字段: {invalid_fields}，可用字段: {available_fields}")

        # 确保保留股票代码和日期
        selected_fields = fields.copy()
        if 'ts_code' not in selected_fields:
            selected_fields.insert(0, 'ts_code')
        if 'trade_date' not in selected_fields:
            selected_fields.insert(1, 'trade_date')

        df = df[selected_fields]

    # 按日期排序
    if 'trade_date' in df.columns:
        df = df.sort_values('trade_date')

    # 限制返回的记录数量
    if count and count > 0:
        df = df.tail(count)

    return df


# 用来获取标的列表
def get_all_securities(date=None):
    import pandas as pd

    # 读取中证500成分股数据
    # 数据文件路径
    file_path = r"D:\read\task\中证500成分股,单一股票数据.csv"

    df = pd.read_csv(file_path, dtype=str)
    stock_codes = df['con_code'].unique()
    print(stock_codes)
    # 添加返回语句
    return stock_codes


"""
修改上面的先
"""


class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.stock_data = self._load_data()
        self.dates = pd.DatetimeIndex(self.stock_data.index.unique(level=0)).sort_values()

    def get_previous_trading_day(self, current_date):
        """获取当前日期的上一个有效交易日"""
        current_date = pd.to_datetime(current_date)
        # 获取所有小于当前日期的交易日并排序
        previous_days = self.dates[self.dates < current_date]
        if len(previous_days) == 0:
            return None  # 没有上一个交易日
        return previous_days[-1]  # 返回最近的一个交易日

    def _load_data(self):
        """加载并预处理pickle数据"""
        df = pd.read_pickle(self.file_path)

        date_column = 'trade_date'
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')  # 强制转换，错误值设为NaT

        invalid_dates = df[date_column].isna().sum()
        if invalid_dates > 0:
            print(f"警告：检测到{invalid_dates}条无效日期记录，已自动删除")
        df = df.dropna(subset=[date_column])

        # 处理股票代码（假设pickle中股票代码字段为'ts_code'，如'000009.SZ'）
        # 提取纯数字部分并补前导零至6位（去除后缀如.SZ/.SH）
        code_column = 'ts_code'
        df[code_column] = df[code_column].astype(str).str.split('.').str[0].str.zfill(6)

        column_mapping = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close'
        }
        df.rename(columns=column_mapping, inplace=True)

        # 设置索引（日期+股票代码）
        return df.set_index([date_column, code_column])
    def get_stock_data(self):
        """回测引擎需要的：获取所有股票数据（用于提取日期列表）"""
        df = self.stock_data.reset_index()  # 此时列：trade_date, ts_code, close, open...
        return df.set_index('trade_date')  # 单索引：trade_date（日期）
    def get_single_day_data(self, date):
        """回测引擎需要的：获取某一天所有股票的收盘价"""
        date = pd.to_datetime(date)
        if date not in self.stock_data.index.levels[0]:
            return pd.Series([np.nan], index=[None])  # 无数据日期返回NaN
        day_data = self.stock_data.loc[date]
        return day_data['close']  # 返回 Series：index=股票代码，value=收盘价



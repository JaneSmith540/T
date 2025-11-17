import pandas as pd
import logging


def get_index_price(start_date, end_date, fields):
    """获取中证500指数价格数据"""
    try:
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
            logging.warning(f"[{start_date} to {end_date}] 无法获取指数数据（日期范围或格式错误）")

        return df

    except FileNotFoundError:
        print(f"错误：找不到文件 C:\\Users\\chanpi\\Desktop\\task\\中证500指数_201601-202506.csv")
        return pd.DataFrame()
    except Exception as e:
        print(f"运行出错：{e}")
        return pd.DataFrame()


# 测试代码
if __name__ == "__main__":
    # 测试用例1：正常情况
    print("=== 测试1：正常情况 ===")
    result1 = get_index_price("2020-01-01", "2020-01-31", ["trade_date", "close"])
    print(f"返回数据行数: {len(result1)}")
    if not result1.empty:
        print(f"列名: {result1.columns.tolist()}")
        print(f"日期范围: {result1['trade_date'].min()} 到 {result1['trade_date'].max()}")

    # 测试用例2：无数据情况
    print("\n=== 测试2：未来日期（应无数据） ===")
    result2 = get_index_price("2030-01-01", "2030-01-31", ["trade_date", "close"])
    print(f"返回数据行数: {len(result2)}")

    # 测试用例3：不存在的字段
    print("\n=== 测试3：不存在的字段 ===")
    result3 = get_index_price("2020-01-01", "2020-01-10", ["trade_date", "nonexistent_field"])
    if not result3.empty:
        print(f"实际返回的列: {result3.columns.tolist()}")
import pandas as pd


class Log:
    @staticmethod
    def info(msg):
        print(f"[INFO] {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}")

    @staticmethod
    def error(msg):
        print(f"[ERROR] {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} - {msg}")


log = Log()  # 实例化log，供策略调用



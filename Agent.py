# 博弈高手
import numpy as np
import pandas as pd
from tqdm import tqdm,trange
# 马尔可夫环境下的判断机器
class Agent():
    def __init__(self, Epsilon, Alpha):
        self.value = np.zeros((3, 3, 3))#因子状态格处置：
        """
        #
        #
        一共27个储存
        3 @ 3 @ 3
        """
        self.Epsilon = Epsilon
        self.Alpha = Alpha


    def decide(self , state):  #返回动作
        if np.random.binomial(1, self.Epsilon):
            random_bit = np.random.binomial(1, 0.5)
            return random_bit  # 结果要么是 0，要么是 1
        else:
            Value = self.value[tuple(state)]
            return Value > 0


# 离散简化智能体环境


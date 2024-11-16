import numpy as np
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sympy.core.random import random

import random
import torch

class CausalForestModel:
    def __init__(self, input_dim, hidden_dim):
        # 输入维度和隐藏层维度不再需要作为直接输入给模型
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 使用econml中的CausalForestDML
        self.model = CausalForestDML(
            model_t=RandomForestClassifier(n_estimators=100),
            model_y=RandomForestRegressor(n_estimators=100),
            n_jobs=-1,
            discrete_treatment=True,
            random_state= int(torch.randint(3, 5, (1,))[0])
        )
    def fit(self, X, T, Y):
        # X: 特征, T: 处理组指示变量, Y: 目标变量
        self.model.fit(Y, T, X=X)

    def predict(self, X):
        # 对给定输入进行预测，返回处理组和对照组的效果
        # 处理组的预测
        return self.model.effect(X)



# 示例用法
# if __name__ == "__main__":
#     # 模拟一些数据
#     np.random.seed(42)
#     n_samples, n_features = 1000, 20
#     X = np.random.randn(n_samples, n_features)
#     T = np.random.randint(0, 2, n_samples)  # 处理组
#     Y = X[:, 0] + 2 * T + np.random.randn(n_samples)  # 目标变量
#
#     # 训练CausalForest模型
#     model = CausalForestModel(input_dim=n_features, hidden_dim=50)
#     model.fit(X, T, Y)
#
#     # 进行预测
#     ite = model.predict(X)
#
#     # 打印预测结果
#     print("Treatment Outcome:", ite[:5])
#     # print("Control Outcome:", control_outcome[:5])
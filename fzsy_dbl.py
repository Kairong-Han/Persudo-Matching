import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
import random
import tqdm
from sklearn.neighbors import KDTree
from torch.utils.data import DataLoader, TensorDataset
import time
# 设置日志配置
import sys
import argparse
from sklift.metrics import qini_auc_score


# 创建解析器
parser = argparse.ArgumentParser(description='处理命令行参数示例')
parser.add_argument('--n', type=int, help='n个数据点', required=True)
parser.add_argument('--r', type=int, help='融合比例', required=True)
parser.add_argument('--l', type=str, help='是否从已有的文件中加载', required=True)
parser.add_argument('--t', type=int, help='第几次训练', required=False,default=0)
# parser.add_argument('--verbose', action='store_true', help='输出详细信息')
args = parser.parse_args()

plt.switch_backend('TkAgg')
is_load = True if args.l == 'y' else False
n = args.n
rate = args.r


seed = 55
log_file = 'log.txt'
local_time = time.localtime()
formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time)
# with open(log_file,'a+') as f:
#     f.write(formatted_time+"\n")
#     f.write(f'参数：n : {n} load : {is_load} rate : {rate} 次数：{args.t}')
#     f.write('\n')

print("当前本地时间:", formatted_time)
random.seed(seed)
np.random.seed(seed)
# torch.manual_seed(seed)

def F(X,U,rate,is_obs=False):
    #引入选择偏差，根据X的数值选择T
    p_T = None
    if is_obs:
        a = np.random.uniform(0, 1, 8)  # X 的系数向量
        b = np.random.uniform(0, 0.25, 6)  # X 的系数向量
        c = np.random.poisson(10, 10)  # X 的系数向量
        linear_combination = X[:,:8].dot(a) + 0.01 * (X[:,6:12] ** 2).dot(b)
        mean = np.mean(linear_combination)
        std = np.std(linear_combination)
        scaled_arr = (linear_combination - mean) / std
        scaled_arr = scaled_arr*2 + 1
        # scaled_arr = (linear_combination - arr_min) / (arr_max - arr_min) * (1 - (-1))
        scaled_arr[scaled_arr < -1.5] = -1.5
        scaled_arr[(scaled_arr>=1) & (scaled_arr < 1.5)] = 1.2
        scaled_arr[scaled_arr > 1.5] = 2.5
        # scaled_arr = X[:,0]
        # scaled_arr[scaled_arr > 50] = -1.5
        p_T = 1 / (2.5 + scaled_arr)
    else:
        a = np.random.uniform(0, 1, X.shape[1])  # X 的系数向量
        b = np.random.uniform(-1, 0.25, X.shape[1])  # X 的系数向量
        c = np.random.poisson(10, X.shape[1])  # X 的系数向量
        linear_combination = X.dot(a) + 0.01 * (X ** 2).dot(b) - 0.0001 * (X ** 3).dot(c)
        mean = np.mean(linear_combination)
        std = np.std(linear_combination)
        scaled_arr = (linear_combination - mean) / std
        scaled_arr = 1.2*scaled_arr
        scaled_arr[scaled_arr < -1] = -1
        scaled_arr[scaled_arr > 1] = 5
        p_T = 1 / (2 + scaled_arr)
    # 通过概率生成 T
    T_selected = np.random.binomial(1, p_T,X.shape[0])
    return T_selected

a_1 = np.random.uniform(-0.75, 1, 20)  # X 的系数向量
a_2 = np.random.uniform(-0.25, 0.75, 20)  # X 的系数向量
a_3 = np.random.uniform(-0.5, 1, 20)  # X 的系数向量
a_4 = np.random.uniform(-0.25, 0.25, 20)  # X 的系数向量

def tau(X):
    return 1 + 0.1*  X.dot(a_1)  + 0.001*(X**2).dot(a_2)

def P(Y_0,Y_1,t=100):
    print(f"{np.sum((Y_1>t)&(Y_0<t))}")
    print(f"{np.sum((Y_1>t)&(Y_0>t))}")
    print(f"{np.sum((Y_1<t)&(Y_0<t))}")
    print(f"{np.sum((Y_1<t)&(Y_0>t))}")

def Generate_Y(X,U,T,n):
    global  a_1,a_2,a_3
    eps = np.random.normal(0,1,n)
    # Y_continuous = 1 + T + -0.1 * X.dot(a_1).T * T + 0.0001 * (X**2).dot(a_2).T * T + eps - 0.2 * X.dot(a_3) + 0.0001 * (X**2).dot(a_4) # 连续值的Y
    Y_continuous = 1 + T + 0.10 * X.dot(a_1).T * T + 0.001 * (X**2).dot(a_2).T * T + eps - 0.050 * X.dot(a_3) + 0.000050 * (X**2).dot(a_4)# 连续值的
    T1 = np.zeros(n)
    Y_0 = 1 + T1 + 0.10 * X.dot(a_1).T * T1 + 0.001 * (X**2).dot(a_2).T * T1 + eps - 0.050 * X.dot(a_3) + 0.000050 * (X**2).dot(a_4)# 连续值的Y
    T1 = np.ones(n)
    Y_1 = 1 + T1 + 0.10 * X.dot(a_1).T * T1 + 0.001 * (X**2).dot(a_2).T * T1 + eps - 0.050 * X.dot(a_3) + 0.000050 * (X**2).dot(a_4)# 连续值的Y
    P(Y_0, Y_1, t=20)
    Y = (Y_continuous > 20).astype(int)  # 将Y转换为二元值（大于0.5时取1，否则取0)
    # P(Y_0, Y_1,20)
    return Y

# 生成仿真数
def generate_pareto_distribution(alpha, size):
    # 生成Pareto分布，alpha为形状参数
    return (np.random.pareto(alpha, size) + 1)  # +1是为了平移使最小值不为0

def generate_rct_features(n_samples=1000):
    """
    模拟RCT实验中的高维特征，包含均匀分布、高斯分布、泊松分布，
    以及通过线性和非线性组合生成的新特征。

    参数:
    n_samples: 样本数量

    返回:
    X: 包含20个特征的特征矩阵 (n_samples, 20)
    """
    # 1. 生成基础特征（均匀分布、高斯分布、泊松分布等）
    X_uniform = np.random.uniform(low=0, high=100, size=(n_samples, 5))  # 5个均匀分布特征
    X_gaussian = np.random.normal(loc=50, scale=10, size=(n_samples, 5))  # 5个高斯分布特征
    X_poisson = np.random.poisson(lam=10, size=(n_samples, 4))  # 3个泊松分布特征
    # 2. 更复杂的特征（模拟现实中更复杂的分布，可以用某些非线性分布模拟）
    # 使用混合高斯分布和双峰分布
    X_mixture = np.concatenate([np.random.normal(-2, 0.5, size=(n_samples // 2, 2)),
                                np.random.normal(3, 1.0, size=(n_samples // 2, 2))], axis=0)
    np.random.shuffle(X_mixture)  # 打乱次序以避免模式太明显
    # 3. 组合特征：线性组合与非线性组合
    # 线性组合
    X_linear_combination = 0.5 * X_uniform[:, 0] + 0.3 * X_gaussian[:, 1] + 0.2 * X_poisson[:, 0]
    X_linear_combination = X_linear_combination.reshape(-1, 1)  # 转为列向量
    # 非线性组合
    X_nonlinear_combination = np.sin(X_uniform[:, 2]) + np.log(1 + X_poisson[:, 1]) - X_gaussian[:, 3] ** 2
    X_nonlinear_combination = X_nonlinear_combination.reshape(-1, 1)  # 转为列向量
    X_long = generate_pareto_distribution(4, (n_samples,1))
    X_Ben = np.random.binomial(100, 0.6,(n_samples,1))
    # 4. 将所有特征拼接成最终的特征矩阵
    X = np.hstack([X_uniform, X_gaussian, X_poisson, X_mixture, X_linear_combination, X_nonlinear_combination,X_Ben,X_long])
    return X

# 生成模拟的特征矩阵，包含1000个样本和20个特征
def draw_all_plt(X, T, Y, T_bias,Y_bias, T_obs,Y_obs,X_obs):
    plt.clf()
    X_treatment = X[T==1]
    X_control = X[T==0]
    plt.figure(figsize=(20, 10))
    # 绘制 X_1 在不同 T 下的密度分布
    for i in range(X.shape[1]):
        plt.subplot(3, X.shape[1], i+1)
        sns.kdeplot(X_treatment[:, i], label='T=1 (Treatment)', color='blue', fill=True)
        sns.kdeplot(X_control[:, i], label='T=0 (Control)', color='red', fill=True)
        # plt.title(f'Density of X_{i} under Different T')
        # plt.xlabel(f'X_{i}')
        # plt.ylabel('Density')
        plt.legend()

    X_treatment = X[T_bias==1]
    X_control = X[T_bias==0]
    for i in range(X.shape[1]):
        plt.subplot(3, X.shape[1], X.shape[1]+i+1)
        sns.kdeplot(X_treatment[:, i], label='T=1 (Treatment)', color='blue', fill=True)
        sns.kdeplot(X_control[:, i], label='T=0 (Control)', color='red', fill=True)
        # plt.title(f'Density of X_{i} under Different T')
        # plt.xlabel(f'X_{i}')
        # plt.ylabel('Density')
        plt.legend()

    X_treatment = X_obs[T_obs==1]
    X_control = X_obs[T_obs==0]
    for i in range(X.shape[1]):
        plt.subplot(3, X.shape[1], 2*X.shape[1]+i+1)
        sns.kdeplot(X_treatment[:, i], label='T=1 (Treatment)', color='blue', fill=True)
        sns.kdeplot(X_control[:, i], label='T=0 (Control)', color='red', fill=True)
        # plt.title(f'Density of X_{i} under Different T')
        # plt.xlabel(f'X_{i}')
        # plt.ylabel('Density')
        plt.legend()
    # 调整布局并显示图像
    # plt.tight_layout()
    plt.show()

def generate_data(n):
    # X的两个维度：X1来自高斯分布，X2来自均匀分布
    X = generate_rct_features(n_samples=n)
    # T来自二项分布，p=0.5
    T = np.random.binomial(1, 0.5, n)
    Y = Generate_Y(X, 0, T, n)

    T_bias = F(X, 0,0.1)
    Y_bias = Generate_Y(X, 0, T_bias, n)

    T_obs = F(X, 0,0.3,True)
    Y_obs = Generate_Y(X, 0,T_obs, n)

    count_Y1 = np.sum(Y == 1)
    count_Y1b = np.sum(Y_bias == 1)
    count_Y1_obs = np.sum(Y_obs == 1)
    print(f"Count of Y=1: {count_Y1} T=1: {np.sum(T == 1)} T=1 and Y=1:{np.sum(Y[T==1] == 1)}")
    print(f"Count of bias Y=1: {count_Y1b} T=1: {np.sum(T_bias == 1)} T=1 and Y=1:{np.sum(Y_bias[T_bias==1] == 1)}")
    print(f"Count of obs Y=1: {count_Y1_obs} T=1: {np.sum(T_obs == 1)} T=1 and Y=1:{np.sum(Y_obs[T_obs==1] == 1)}")
    # draw_all_plt(X, T, Y, T_bias, Y_bias, T_obs, Y_obs)
    return X, T, Y, T_bias,Y_bias, T_obs,Y_obs

def draw_plt(X,T,file_name):
    plt.clf()
    X_treatment = X[T==1]
    X_control = X[T==0]
    plt.figure(figsize=(12, 5))
    # 绘制 X_1 在不同 T 下的密度分布
    for i in range(X.shape[1]):
        plt.subplot(1, X.shape[1], i+1)
        sns.kdeplot(X_treatment[:, i], label='T=1 (Treatment)', color='blue', fill=True)
        sns.kdeplot(X_control[:, i], label='T=0 (Control)', color='red', fill=True)
        # plt.title(f'Density of X_{i} under Different T')
        # plt.xlabel(f'X_{i}')
        # plt.ylabel('Density')
        plt.legend()
    plt.show()

def concatenate_and_normalize(A, B):
    # 1. 沿列方向拼接两个矩阵
    concatenated = np.concatenate((A, B), axis=0)
    # 2. 对拼接后的矩阵进行列归一化
    col_min = concatenated.min(axis=0)
    col_max = concatenated.max(axis=0)
    # 避免分母为零的情况
    col_range = col_max - col_min
    col_range[col_range == 0] = 1  # 若某列最大值等于最小值，将范围设为1，避免除以0
    normalized = (concatenated - col_min) / col_range
    # 3. 重新分割成两个矩阵
    A_normalized = normalized[:A.shape[0], :]
    B_normalized = normalized[A.shape[0]:, :]
    return A_normalized, B_normalized

def Data_fusion(RCT_X,RCT_T,OBS_X,OBS_T,OBS_Y,rate=1):
    avg_u = np.mean(RCT_X,axis=0)
    bias_0 = avg_u - np.mean(RCT_X[RCT_T==0],axis=0)
    bias_1 = avg_u - np.mean(RCT_X[RCT_T==1],axis=0)
    bias = [bias_0,bias_1]
    select_row = []
    inf = 99999
    for T in [0,1]:
        persu = RCT_X[RCT_T == T] + (1+1/rate)*bias[T]
        OBS_temp = OBS_X[OBS_T == T]
        Persu_norm,OBS_norm = concatenate_and_normalize(persu,OBS_temp)
        tree = KDTree(OBS_norm, leaf_size=40)
        index_set = []
        for item in tqdm.tqdm(Persu_norm):
            distances,indices = tree.query([item],rate)
            index_set.extend( indices.flatten() )
        print(f'before {len(index_set)}')
        indices = set(index_set)
        print(f'after {len(indices)}')

        # min_window = [inf]*rate
            # min_window_ind = [0]*rate
            # for obs_i,obs in enumerate(OBS_norm):
            #     if obs_i not in index_set:
            #         dis = np.linalg.norm(item - obs)
            #         if dis < max(min_window):
            #             index = np.argmax(min_window)
            #             min_window[index] = dis
            #             min_window_ind[index]=obs_i
            # index_set.update(set(min_window_ind))
        select = list(indices)
        select_X = OBS_X[OBS_T == T][select]
        select_T = [T]*len(select)
        select_Y = OBS_Y[OBS_T == T][select]
        select_row.append([select_X,select_T,select_Y])

    print(f'selected number : {len(select_row[0][0])+len(select_row[1][0])}')
    return  np.concatenate((select_row[0][0], select_row[1][0]), axis=0),\
            np.concatenate((select_row[0][1], select_row[1][1]), axis=0),\
            np.concatenate((select_row[0][2], select_row[1][2]), axis=0)


def Data_fusion_random(RCT_X,RCT_T,OBS_X,OBS_T,OBS_Y,rate=1):
    select_row = []
    for T in [0,1]:
        rct = RCT_X[RCT_T == T]
        obs = OBS_X[OBS_T == T]
        num_samples = len(rct)
        # 当 rct 的长度大于 obs 的长度时，选中全部 obs
        if num_samples > len(obs):
            select = list(range(len(obs)))  # 选择所有的索引
        else:
            select = random.sample(range(len(obs)), num_samples)
        select_X = OBS_X[OBS_T == T][select]
        select_T = [T]*len(select)
        select_Y = OBS_Y[OBS_T == T][select]
        select_row.append([select_X,select_T,select_Y])
    print(f'selected number : {len(select_row[0][0])+len(select_row[1][0])}')
    return  np.concatenate((select_row[0][0], select_row[1][0]), axis=0),\
            np.concatenate((select_row[0][1], select_row[1][1]), axis=0),\
            np.concatenate((select_row[0][2], select_row[1][2]), axis=0)


def train_model(model, x_train, y_train, t_train,optimizer,scheduler,num_epochs=20):
    train_dataset = TensorDataset(x_train, y_train, t_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
    model.train()
    for epoch in range(num_epochs):
        for batch_x, batch_y, batch_t in train_loader:
            optimizer.zero_grad()
            # 前向传播：通过模型获取处理和控制组的预测值
            treatment_outcome, control_outcome = model(batch_x)
            # 根据 t_train 的值选择相应的预测输出
            pred_y = torch.where(batch_t == 1, treatment_outcome.squeeze(), control_outcome.squeeze())
            # 计算损失
            loss = criterion(pred_y.squeeze(), batch_y)
            # 反向传播和优化
            loss.backward()
            optimizer.step()
        # 学习率调度器更新
        # scheduler.step()
        # 每隔10个epoch打印一次损失
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
# 生成 1000 个数据点
class TARNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TARNet, self).__init__()
        # 共享表示网络
        dropout_rate = 0.05
        self.shared_rep = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),  # 添加 Dropout
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)  # 添加 Dropout
        )

        # 处理组的输出层
        self.treatment_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),  # 添加 Dropout
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),  # 添加 Dropout
            nn.Linear(hidden_dim, 1)
        )

        # 对照组的输出层
        self.control_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),  # 添加 Dropout
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),  # 添加 Dropout
            nn.Linear(hidden_dim, 1)
        )

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # 共享表示
        shared_rep = self.shared_rep(x)

        # 针对处理组和对照组的预测
        treatment_outcome = self.sigmoid(self.treatment_head(shared_rep))
        control_outcome = self.sigmoid(self.control_head(shared_rep))

        return treatment_outcome, control_outcome

def qini_coefficient(y_true, t, uplift, n_bins=1000):
    """
    计算 Qini 系数。
    参数:
    y_true -- 真实的 Y 值 (0 或 1)
    t -- 处理指示 (0 或 1)
    uplift -- 模型预测的因果效应 (HTE)
    n_bins -- 划分的分区数，默认是10
    返回:
    qini_coef -- 计算得到的 Qini 系数
    """

    # 计算个体的 uplift 得分并排序
    order = np.argsort(-uplift)  # 按降序排序
    y_true = y_true[order]
    t = t[order]

    # 分区
    bin_size = len(y_true) // n_bins
    cum_y_treat = np.cumsum(y_true * t)  # 处理组的累计效果
    cum_y_control = np.cumsum(y_true * (1 - t))  # 对照组的累计效果
    t_t_num =  np.cumsum(t)
    t_c_num =  np.cumsum(1 - t)
    t_t_num[t_t_num == 0] = 1
    t_c_num[t_c_num == 0] = 1
    normalization_factor = t_t_num/t_c_num
    # print(normalization_factor[:5])
    # 计算 Qini 曲线上的累积增益
    qini_curve = cum_y_treat - cum_y_control * normalization_factor
    random_curve = np.linspace(0, qini_curve[-1], len(qini_curve))

    # Qini 系数为实际 Qini 曲线与随机曲线之间的面积
    qini_coef = np.trapz(qini_curve - random_curve)
    # print(qini_coef)

    plt.figure(figsize=(8, 6))
    plt.plot(np.linspace(0, 1, len(qini_curve)), qini_curve, label='Qini Curve', color='blue')
    plt.plot(np.linspace(0, 1, len(random_curve)), random_curve, label='Random Curve', linestyle='--', color='red')
    plt.title('Qini Curve')
    plt.xlabel('Fraction of Population')
    plt.ylabel('Cumulative Gain')
    plt.legend()
    plt.grid(True)
    plt.show()
    return qini_coef

def qini_mape_copc(y_true, t, uplift):
    y_true = np.array(y_true)
    t = np.array(t)
    order = np.argsort(-uplift)  # 按降序排序
    y_true = y_true[order]
    t = t[order]
    # 分区
    cum_y_treat = np.cumsum(y_true * t)  # 处理组的累计效果
    cum_y_control = np.cumsum(y_true * (1 - t))  # 对照组的累计效果
    cum_uplift = np.cumsum(uplift[order] * t)
    cum_uplift[cum_uplift == 0] = 1
    t_t_num =  np.cumsum(t)
    t_c_num =  np.cumsum(1 - t)
    t_t_num[t_t_num == 0] = 1
    t_c_num[t_c_num == 0] = 1
    normalization_factor = t_t_num/t_c_num
    qini_curve = cum_y_treat - cum_y_control * normalization_factor
    qini_curve[qini_curve==0] = 1
    mape = np.abs(qini_curve-cum_uplift)/np.abs(qini_curve)
    cpoc = qini_curve/cum_uplift
    return np.mean(mape),np.mean(cpoc)

X, T, Y, T_bias,Y_bias,_,_ = generate_data(n)

# draw_plt(X,T,'golden_rct')
# draw_plt(X,T_bias,'bias_rct')

# 分割数据集为训练集（80%）和测试集（20%）
# 转换为 PyTorch 张量

X_obs,T_obs,Y_obs,T_g,Y_g = None,None,None,T,Y
X_obs, T_g, Y_g, _, _ ,T_obs,Y_obs = generate_data(100*n)


X_train, X_test, T_train, T_test, Y_train, Y_test, Tb_train, Tb_test, Yb_train, Yb_test = train_test_split(X, T, Y,T_bias,Y_bias, test_size=0.2, random_state=42)
X_obs_train,X_obs_test, T_g_train,T_g_test, Y_g_train,Y_g_test,T_obs_train,T_obs_test,Y_obs_train,Y_obs_test = train_test_split(X_obs, T_g, Y_g ,T_obs,Y_obs, test_size=0.2, random_state=42)

X_obs, T_g, Y_g,T_obs,Y_obs = X_obs_train, T_g_train, Y_g_train,T_obs_train,Y_obs_train

X_train_tensor = torch.FloatTensor(X_train)
X_test_tensor = torch.FloatTensor(X_test)
T_train_tensor = torch.FloatTensor(T_train)
T_test_tensor = torch.FloatTensor(T_test)
Y_train_tensor = torch.FloatTensor(Y_train)
Y_test_tensor = torch.FloatTensor(Y_test)
Tb_train_tensor = torch.FloatTensor(Tb_train)
Tb_test_tensor = torch.FloatTensor(Tb_test)
Yb_train_tensor = torch.FloatTensor(Yb_train)
Yb_test_tensor = torch.FloatTensor(Yb_test)
X_obs = torch.FloatTensor(X_obs_test)
T_g = torch.FloatTensor(T_g_test)
Y_g = torch.FloatTensor(Y_g_test)

input_dim = 20  # X 有两个维度
hidden_dim = 100
model_b = TARNet(input_dim, hidden_dim)
criterion = nn.BCELoss()
optimizer_b = optim.Adam(model_b.parameters(), lr=0.0005)
scheduler_b = optim.lr_scheduler.StepLR(optimizer_b, step_size=2, gamma=0.5)
train_model(model_b, X_train_tensor, Yb_train_tensor, Tb_train_tensor,optimizer_b, scheduler_b)
model_b.eval()
with torch.no_grad():
    treatment_outcome_test, control_outcome_test = model_b(X_test_tensor)
HTE_test = (treatment_outcome_test - control_outcome_test).squeeze().numpy()
qini_coef = qini_auc_score(y_true=Yb_test, uplift=HTE_test, treatment=Tb_test)
mape, copc = qini_mape_copc(Yb_test, Tb_test, HTE_test)
with open(log_file, 'a+') as f:
    f.write(f'{qini_coef}, {mape}, {copc}')
    f.write('\n')
print(f"bias Qini coefficient: {qini_coef:.4f} mape_copc {mape} {copc}")
# 大规模黄金RCT上的测试
model_b.eval()
with torch.no_grad():
    treatment_outcome_test, control_outcome_test = model_b(X_obs)
HTE_test = (treatment_outcome_test - control_outcome_test).squeeze().numpy()
qini_coef = qini_auc_score(y_true=Y_g, uplift=HTE_test, treatment=T_g)
mape, copc = qini_mape_copc(Y_g, T_g, HTE_test)
with open(log_file, 'a+') as f:
    f.write(f'{qini_coef}, {mape}, {copc}')
    f.write('\n')
print(f"ground truth Qini coefficient: {qini_coef:.4f} mape_copc {qini_mape_copc(Y_g, T_g, HTE_test)}")


for rate in [1,3,5]:
    for mode in ['ours','random']:
        if not is_load:
            if mode == 'ours':
                select_x,select_t,select_y = Data_fusion(X_train,Tb_train,X_obs_train,T_obs_train,Y_obs_train,rate=rate)
            else:
                select_x,select_t,select_y = Data_fusion_random(X_train,Tb_train,X_obs_train,T_obs_train,Y_obs_train,rate=rate)
            X_train_fusion = np.concatenate((X_train,select_x),axis=0)
            Y_train_fusion = np.concatenate((Yb_train,select_y),axis=0)
            T_train_fusion = np.concatenate((Tb_train,select_t),axis=0)
            np.savetxt(f'X_train_fusion_{rate}_{mode}.txt', X_train_fusion)
            np.savetxt(f'Y_train_fusion_{rate}_{mode}.txt', Y_train_fusion)
            np.savetxt(f'T_train_fusion_{rate}_{mode}.txt', T_train_fusion)
        else:
            X_train_fusion = np.loadtxt(f'X_train_fusion_{rate}_{mode}.txt')
            Y_train_fusion = np.loadtxt(f'Y_train_fusion_{rate}_{mode}.txt')
            T_train_fusion = np.loadtxt(f'T_train_fusion_{rate}_{mode}.txt')

        # draw_plt(X_train,T_train,'golden_rct')
        # draw_plt(X_train[:,2:5],Tb_train,'bias_rct')
        # draw_plt(X_obs_train[:,2:5],T_obs_train,'obs')
        # draw_plt(X_train_fusion[:,2:5],T_train_fusion,'fusion_rct')
        # draw_all_plt(X_train,T_train,_,Tb_train,_,T_train_fusion,_,X_train_fusion)

        T_train_fusion_tensor = torch.FloatTensor(T_train_fusion)
        Y_train_fusion_tensor = torch.FloatTensor(Y_train_fusion)
        X_train_fusion_tensor = torch.FloatTensor(X_train_fusion)


        model_f = TARNet(input_dim, hidden_dim)
        # 定义损失函数和优化器

        optimizer_f = optim.Adam(model_f.parameters(), lr=0.0005)

        scheduler_f = optim.lr_scheduler.StepLR(optimizer_f, step_size=2, gamma=0.5)

        # 训练模型
        train_model(model_f, X_train_fusion_tensor, Y_train_fusion_tensor, T_train_fusion_tensor,optimizer_f,scheduler_f)

        # TARNet 模型预测处理组和对照组潜在结果

        model_f.eval()
        with torch.no_grad():
            treatment_outcome_test_f, control_outcome_test_f = model_f(X_test_tensor)

        # 计算每个个体的异质性因果效应 (HTE)
        HTE_test_f = (treatment_outcome_test_f - control_outcome_test_f).squeeze().numpy()

        # 计算 Qini 系数
        #在黄金上做测试作为ground truth
        #
        # qini_coef = qini_auc_score(y_true=Y_test, uplift=HTE_test, treatment=T_test)
        # print(f"ground truth Qini coefficient: {qini_coef:.4f} mape_copc {qini_mape_copc(Y_test, T_test, HTE_test)}")
        # # qini_coef = qini_auc_score(y_true=Y_test, uplift=HTE_test_g, treatment=T_test)
        # # print(f"ground truth Qini coefficient by golden: {qini_coef:.4f}  mape_copc {qini_mape_copc(Y_test, T_test, HTE_test_g)}")
        # qini_coef = qini_auc_score(y_true=Y_test, uplift=HTE_test_f, treatment=T_test)
        # print(f"ground truth Qini coefficient by fusion: {qini_coef:.4f}  mape_copc {qini_mape_copc(Y_test, T_test, HTE_test_f)}")

        #在有偏上做测试作为真实场景中得到的

        # with open(log_file,'a+') as f:
        #     f.write(f'有偏的RCT')
        #     f.write('\n')


        # qini_coef = qini_auc_score(y_true=Yb_test, uplift=HTE_test_g, treatment=Tb_test)
        # mape,copc = qini_mape_copc(Yb_test, Tb_test, HTE_test_g)
        # with open(log_file,'a+') as f:
        #     f.write(f'{qini_coef}, {mape}, {copc}')
        #     f.write('\n')


        # print(f"bias Qini coefficient by golden: {qini_coef:.4f} mape_copc {qini_mape_copc(Yb_test, Tb_test, HTE_test_g)}")
        qini_coef = qini_auc_score(y_true=Yb_test, uplift=HTE_test_f, treatment=Tb_test)
        mape,copc = qini_mape_copc(Yb_test, Tb_test, HTE_test_f)
        with open(log_file,'a+') as f:
            f.write(f'{qini_coef}, {mape}, {copc}')
            f.write('\n')

        print(f"bias Qini coefficient by fusion {mode}_{rate}: {qini_coef:.4f} mape_copc {qini_mape_copc(Yb_test, Tb_test, HTE_test_f)}")



        # 大规模黄金RCT上的测试
        model_f.eval()
        with torch.no_grad():
            treatment_outcome_test_f, control_outcome_test_f = model_f(X_obs)

        # 计算每个个体的异质性因果效应 (HTE)

        HTE_test_f = (treatment_outcome_test_f - control_outcome_test_f).squeeze().numpy()

        from sklift.metrics import qini_auc_score

        # with open(log_file,'a+') as f:
        #     f.write(f'大规模黄金RCT')
        #     f.write('\n')
        # 计算 Qini 系数
        #在黄金上做测试作为ground truth

        qini_coef = qini_auc_score(y_true=Y_g, uplift=HTE_test_f, treatment=T_g)
        mape,copc = qini_mape_copc(Y_g, T_g, HTE_test_f)
        with open(log_file,'a+') as f:
            f.write(f'{qini_coef}, {mape}, {copc}')
            f.write('\n')

        print(f"ground truth Qini coefficient by fusion {mode}_{rate}: {qini_coef:.4f} mape_copc {qini_mape_copc(Y_g, T_g, HTE_test_f)}")
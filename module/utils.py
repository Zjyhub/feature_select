import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score

bar_format = '{desc}: {n}/{total} {elapsed} [{remaining},{rate_fmt}] {postfix} {percentage:3.1f}% |{bar}|'

# 返回sigmoid函数值
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 返回tanh函数值的绝对值
def tanh(x):
    return np.abs(np.tanh(x))


# 返回反转解
def obl(x):
    return 1 - x


def fitness(alpha, beta, dimension, train_X, y, x, knn, k=5):
    """
    计算适应度函数值

    参数:
    alpha: 惩罚系数，用于控制分类准确率的权重
    beta: 惩罚系数，用于控制特征数量的权重
    dimension: 特征数量
    train_X: 特征矩阵，形状为 (样本数量, 特征数量)
    y: 目标类别标签，形状为 (样本数量,)
    x: 特征选择向量，形状为 (特征数量,)，其中1表示选择该特征，0表示不选择
    knn: KNN分类器对象
    k: kfold交叉验证的k值，默认值为10
    """
    train_X = train_X.iloc[:, x == 1]

    # 如果选择的特征数量为0，则返回正无穷
    if train_X.shape[1] == 0:
        return float("inf")

    # 随机选择数据集的70%作为训练集，30%作为测试集
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 使用kfold交叉验证划分数据集
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    err_rates = []

    # 计算每一折的错误率
    for train_index, test_index in kf.split(train_X):
        # 对于训练集和测试集采用k重交叉验证
        X_train_kf, X_test_kf = train_X.iloc[train_index], train_X.iloc[test_index]
        y_train_kf, y_test_kf = y.iloc[train_index], y.iloc[test_index]
        if len(X_train_kf) < 5:
            print(train_index)
        knn.fit(X_train_kf, y_train_kf)  # 训练
        y_pred = knn.predict(X_test_kf)  # 预测
        err_rates.append(1 - accuracy_score(y_test_kf, y_pred))  # 计算错误率

    s = x.sum()  # 计算当前已选择的特征数量
    err_rate = np.mean(err_rates)  # 计算平均错误率
    f = alpha * err_rate + beta * s / dimension  # 计算适应度函数值
    return f.astype(float)

# 保存结果到文件
def save_result(algorithm_name, accuracy_mean, best_solution, best_accuracy, run_times, Dataset):
        # 保存结果追加到文件中,记录格式为: [时间] [算法名称][运行次数][平均准确率][最优解][最佳准确率]
        # 如果文件不存在则创建文件
        if not os.path.exists(f"./output/{algorithm_name}/result"):
            os.makedirs(f"./output/{algorithm_name}/result")

        with open(f"./output/{algorithm_name}/result/{algorithm_name}_{Dataset}.txt", "a") as f:
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{date}] {algorithm_name} run {run_times} times , mean accuracy: {accuracy_mean*100:.2f}%, best solution: {best_solution}, best accuracy: {best_accuracy:.2f}%\n")

# 保存图像
def save_figure(algorithm_name, f_list, run_times, Dataset):
    # f_list:是一个二维数组，每一行代表一次运行的适应度值
    # 计算f_list每一列的中位数
    if not os.path.exists(f"./output/{algorithm_name}/figure"):
        os.makedirs(f"./output/{algorithm_name}/figure")
    median = np.median(f_list, axis=0)

    sample_interval = 20
    median = median[::sample_interval]
    plt.plot(median, label=algorithm_name,marker='o')
    plt.xlabel("iteration")
    plt.ylabel("fitness")
    plt.legend()
    plt.title(f"{algorithm_name} run {run_times} times")
    plt.savefig(f"./output/{algorithm_name}/figure/{algorithm_name}_{Dataset}.png")

# 根据传入路径读取数据，返回特征矩阵X和目标类别标签y
def read_uci_data(path, y_index=0):
    data = pd.read_csv(path, header=None)  # 读取数据

    X = data.iloc[
        :,
        [
            i
            for i in range(data.shape[1])
            if i != y_index and i - data.shape[1] != y_index
        ],
    ]  # 特征矩阵

    # 将X中数据类型为string的列转换为数值类型
    X=X.apply(pd.to_numeric, errors='coerce').fillna(0)

    y = data.iloc[:, y_index]

    # 将y根据类别进行编码
    unique_y = y.unique()
    y = y.apply(lambda x: unique_y.tolist().index(x))
    return X, y

# 编写CEC2017的测试函数
def cec2017_F1(x):
    return np.sum(x**2)  # 返回x的平方和

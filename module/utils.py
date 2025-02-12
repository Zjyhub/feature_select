import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score



# 返回sigmoid函数值
def sigmoid(x):
    return 1/(1+np.exp(-x))

# 返回tanh函数值的绝对值
def tanh(x):
    return np.abs(np.tanh(x))

# 返回反转解
def obl(x):
    return 1-x

def fitness(alpha, beta, dimension, X, y, x, knn, k=5):
    """
    计算适应度函数值

    参数:
    alpha: 惩罚系数，用于控制分类准确率的权重
    beta: 惩罚系数，用于控制特征数量的权重
    dimension: 特征数量
    X: 特征矩阵，形状为 (样本数量, 特征数量)
    y: 目标类别标签，形状为 (样本数量,)
    x: 特征选择向量，形状为 (特征数量,)，其中1表示选择该特征，0表示不选择
    knn: KNN分类器对象
    k: kfold交叉验证的k值，默认值为10
    """
    X = X.iloc[:,x==1]

    # 如果选择的特征数量为0，则返回正无穷
    if X.shape[1] == 0:
        return float('inf')

    # 随机选择数据集的70%作为训练集，30%作为测试集
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 使用kfold交叉验证划分数据集
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    err_rates = []

    # 计算每一折的错误率
    for train_index, test_index in kf.split(X):
        # 对于训练集和测试集采用k重交叉验证
        X_train_kf, X_test_kf = X.iloc[train_index], X.iloc[test_index]
        y_train_kf, y_test_kf = y.iloc[train_index], y.iloc[test_index]
        if len(X_train_kf) < 5:
            print(train_index)
        knn.fit(X_train_kf, y_train_kf) # 训练
        y_pred = knn.predict(X_test_kf) # 预测
        err_rates.append(1 - accuracy_score(y_test_kf, y_pred)) # 计算错误率
        
    s = x.sum() # 计算当前已选择的特征数量
    err_rate = np.mean(err_rates) # 计算平均错误率
    f = alpha * err_rate + beta * s / dimension # 计算适应度函数值
    return f.astype(float)

# 编写CEC2017的测试函数
def cec2017_F1(x):
    return np.sum(x**2) # 返回x的平方和






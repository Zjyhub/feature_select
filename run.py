import pandas as pd
from module.FeatureSelect import *


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
    y = data.iloc[:, y_index]

    # 将y根据类别进行编码
    unique_y = y.unique()
    y = y.apply(lambda x: unique_y.tolist().index(x))
    return X, y


if __name__ == "__main__":
    X, y = read_uci_data("./data/wine/wine.data")
    f = FeatureSelect(X, y)
    # f.compare()  # 比较四种算法
    f.fit_DE_SHADE("wine")  # 运行DE_SHADE算法
    # f.fit_DE_JADE('wine') # 运行DE_JADE算法
    # f.fit_DE('wine') # 运行DE算法
    # f.fit_BPSO('wine') # 运行BPSO算法
    # f.fit_BPSO_Obl('wine') # 运行BPSO_Obl算法

import pandas as pd
from module.FeatureSelect import *

# 数据集列表,格式为(路径, y_index, 名称)
dataset_list = [
    ("./data/wine/wine.data", 0, "wine"),
    ("./data/ionosphere/ionosphere.data", 0, "ionosphere"),
    ("./data/lymphography/lymphography.data", 0, "lymphography"),
    ("./data/zoo/zoo.data", 17, "zoo"),
    ("./data/spambase/spambase.data", 57, "spambase"),
]


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
    # for dataset in dataset_list:
    #     X, y = read_uci_data(dataset[0], dataset[1])
    #     f = FeatureSelect(X, y)
    #     print(f"当前正在对{dataset[2]}数据集进行特征选择:")
    #     f.fit_BPSO(dataset[2])
    #     f.fit_BPSO_Obl(dataset[2])
    #     f.fit_DE(dataset[2])
    #     f.fit_DE_JADE(dataset[2])
    #     f.fit_DE_SHADE(dataset[2])
    #     f.fit_DE_LSHADE(dataset[2])
    #     f.fit_DE_RL_LSHADE(dataset[2])
    #     f.compare()

    X, y = read_uci_data("./data/wine/wine.data")
    f = FeatureSelect(X, y)
    # f.fit_BPSO('wine') # 运行BPSO算法
    # f.fit_BPSO_Obl('wine') # 运行BPSO_Obl算法
    f.fit_DE("wine")  # 运行DE算法
    f.fit_DE_JADE("wine")  # 运行DE_JADE算法
    f.fit_DE_SHADE("wine")  # 运行DE_SHADE算法
    f.fit_DE_LSHADE("wine")  # 运行DE_LSHADE算法
    f.fit_DE_RL_LSHADE("wine")  # 运行DE_RL_LSHADE算法
    f.compare()  # 比较四种算法的性能

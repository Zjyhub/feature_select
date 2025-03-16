'''
Description: 
Author: Zjy
Date: 2025-03-03 23:45:26
LastEditTime: 2025-03-16 16:22:22
version: 1.0
'''
from module.utils import read_uci_data
from module.FeatureSelect import *

# 数据集列表,格式为(路径, y_index, 名称)
Dataset_list = [
    ("./data/wine/wine.data", 0, "wine"),
    # ("./data/ionosphere/ionosphere.data", -1, "ionosphere"),
    # ("./data/lymphography/lymphography.data", 0, "lymphography"),
    # ("./data/zoo/zoo.data", -1, "zoo"),
    # ("./data/spambase/spambase.data", 57, "spambase"),
]

Alorithm_list = ["BPSO", "BPSO_OBL", "DE", "DE_JADE", "DE_SHADE", "DE_LSHADE", "DE_RL_LSHADE"]

# 运行所有的数据集和算法并输出表格
def save_table():
    dataset_accuracy = []
    col_index=[]
    for i in range(len(Dataset_list)):
        col_index.append(Dataset_list[i][2])
        X, y = read_uci_data(Dataset_list[i][0], Dataset_list[i][1])
        fs = FeatureSelect(X, y, Dataset_list[i][2])
        accuracy_list = fs.compare(algorithm_list=Alorithm_list, run_times=20)
        accuracy_list = [f"{accuracy*100:.2f}%" for accuracy in accuracy_list]
        dataset_accuracy.append(accuracy_list)
    df = pd.DataFrame(dataset_accuracy, columns=Alorithm_list, index=col_index)
    print(df.to_csv("./output/accuracy_table.csv"))


if __name__ == "__main__":
    # save_table()
    
    d_index = 0 # 选择数据集
    a_index = 1 # 选择算法
    X, y = read_uci_data(Dataset_list[d_index][0], Dataset_list[d_index][1])
    fs = FeatureSelect(X, y, Dataset_list[d_index][2])
    fs.fit(Alorithm_list[a_index], run_times=1)
    # fs.compare(algorithm_list=Alorithm_list, run_times=2)

'''
Description: 
Author: Zjy
Date: 2025-03-20 20:39:58
LastEditTime: 2025-04-04 18:42:05
version: 1.0
'''
from sklearn.metrics import accuracy_score
from module.utils import read_uci_data,cal_accuracy
from module.FeatureSelect import *

# 数据集列表,格式为(路径, y_index, 名称)
Dataset_list = [
    ("./data/Breast/Breast.data", 0, "Breast"),
    ("./data/wine/wine.data", 0, "wine"),
    ("./data/zoo/zoo.data", -1, "zoo"),
    ("./data/CNAE-9/CNAE-9.data", 0, "CNAE-9"),
    ("./data/hill-valley/Hill_Valley_without_noise.data",-1, "hill-valley"),
    ("./data/lung-cancer/lung-cancer.data", 0, "lung-cancer"),
    ("./data/ionosphere/ionosphere.data", -1, "ionosphere"),
    ("./data/lymphography/lymphography.data", 0, "lymphography"),
    ("./data/madelon/madelon.data", 0, "madelon"),
    ("./data/movement_libras/movement_libras.data", -1, "movement_libras"),
    ("./data/musk1/clean1.data", -1, "musk1"),
    ("./data/semeion/semeion.data", -1, "semeion"),
    ("./data/sonar/sonar.data", -1, "sonar"),
    ("./data/spambase/spambase.data", -1, "spambase"),

]

Alorithm_list = [
    "DE",
    "BPSO",
    "DE_JADE",
    "DE_SHADE",
    "DE_LSHADE",
    "DE_RL",
    "DE_RL_LSHADE",
    "DE_best_2",

    # "DE_DynamicF",
    # "DE_DynamicF_2",
    # "DE_model",
    # "BPSO_OBL",
]


# 运行所有的数据集和算法并输出表格
def save_table():
    dataset_accuracy = []
    dataset_feature_num = []
    row_index = []
    for i in range(len(Dataset_list)):
        row_index.append(Dataset_list[i][2])
        X, y = read_uci_data(Dataset_list[i][0], Dataset_list[i][1])
        fs = FeatureSelect(X, y, Dataset_list[i][2])
        accuracy_list,feature_list=fs.compare(algorithm_list=Alorithm_list, run_times=20)
        accuracy_list.insert(0,cal_accuracy(X, y, np.ones(X.shape[1])))
        feature_list.insert(0,X.shape[1])

        accuracy_list = [f"{accuracy*100:.2f}" for accuracy in accuracy_list]
        feature_list = [f"{f_num:.2f}" for f_num in feature_list]

        dataset_accuracy.append(accuracy_list)
        dataset_feature_num.append(feature_list)
    
    col_index=Alorithm_list.copy()
    col_index.insert(0,"Full")
    df = pd.DataFrame(dataset_accuracy, columns=col_index, index=row_index)
    # df.to_csv("./output/accuracy_table_8.csv")
    df.to_excel("./output/accuracy_table_8.xlsx")

    df = pd.DataFrame(dataset_feature_num, columns=col_index, index=row_index)
    # df.to_csv("./output/feature_num_table_8.csv")
    df.to_excel("./output/feature_num_table_8.xlsx")


if __name__ == "__main__":
    d_index = 2  # 选择数据集
    a_index = 0  # 选择算法
    X, y = read_uci_data(Dataset_list[d_index][0], Dataset_list[d_index][1])
    fs = FeatureSelect(X, y, Dataset_list[d_index][2])
    # fs.fit(Alorithm_list[a_index], run_times=1)
    fs.compare(algorithm_list=Alorithm_list, run_times=1)
    
    # save_table()

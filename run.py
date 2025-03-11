from module.utils import read_uci_data
from module.FeatureSelect import *

# 数据集列表,格式为(路径, y_index, 名称)
Dataset_list = [
    ("./data/wine/wine.data", 0, "wine"),
    ("./data/ionosphere/ionosphere.data", -1, "ionosphere"),
    ("./data/lymphography/lymphography.data", 0, "lymphography"),
    ("./data/zoo/zoo.data", 17, "zoo"),
    ("./data/spambase/spambase.data", 57, "spambase"),
]

Alorithm_list = ["BPSO", "BPSO_Obl", "DE", "DE_JADE", "DE_SHADE", "DE_LSHADE", "DE_RL_LSHADE"]



if __name__ == "__main__":
    
    d_index = 1 # 选择数据集
    a_index = 0 # 选择算法
    X, y = read_uci_data(Dataset_list[d_index][0], Dataset_list[d_index][1])
    fs = FeatureSelect(X, y, Dataset_list[d_index][2])
    fs.fit(Alorithm_list[a_index], run_times=4)

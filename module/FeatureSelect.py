from module.utils import *
from module.BPSO import *
from module.BPSO_OBL import *
from module.DE import *
from module.DE_JADE import *
from module.DE_SHADE import *
from module.DE_LSHADE import *
from module.DE_RL_LSHADE import *
from module.DE_DynamicF import *
from module.DE_model import *
from matplotlib import pyplot as plt


algorithms = [
    "BPSO",
    "BPSO_OBL",
    "DE",
    "DE_JADE",
    "DE_SHADE",
    "DE_LSHADE",
    "DE_RL_LSHADE",
    "DE_DynamicF",
    "DE_model",
]


class FeatureSelect:
    def __init__(self, X, y, Dataset, M=20):
        """
        初始化特征选择对象

        参数:
        X: 特征矩阵，形状为 (样本数量, 特征数量)
        y: 目标类别标签，形状为 (样本数量,)
        Dataset: 数据集名称
        M: 特征选择次数，默认值为1
        """
        self.X = X
        self.y = y
        self.Dataset = Dataset
        self.M = M

        self.algorithm_models = {
            "BPSO": BPSO(self.X, self.y),
            "BPSO_OBL": BPSO_OBL(self.X, self.y),
            "DE": DE(self.X, self.y),
            "DE_JADE": DE_JADE(self.X, self.y),
            "DE_SHADE": DE_SHADE(self.X, self.y),
            "DE_LSHADE": DE_LSHADE(self.X, self.y),
            "DE_RL_LSHADE": DE_RL_LSHADE(self.X, self.y),
            "DE_DynamicF": DE_DynamicF(self.X, self.y),
            "DE_model": DE_model(self.X, self.y),
        }

    def choose_algorithm(self, algorithm_name):
        # 根据算法名称选择对应的算法
        if algorithm_name in algorithms:
            model = self.algorithm_models[algorithm_name]
        else:
            raise ValueError("未知的算法名称")

        return model

    def fit(
        self,
        algorithm_name,  # 算法名称
        run_times=1,  # 运行次数
        is_save=True,  # 是否保存结果
        is_plot=True,  # 是否画图
    ):
        model = self.choose_algorithm(algorithm_name)  # 根据算法名称选择对应的算法

        # 初始化结果列表
        accuracy_list = []
        solution_list = []
        f_list = []

        print(f"{algorithm_name} 对{self.Dataset}数据集的特征选择，开始运行...")
        for i in range(run_times):
            accuracy_list.append(model.fit())
            solution_list.append(model.global_best)
            f_list.append(model.f_best)

        # 根据accuracy_list计算平均准确率，并找到最优解
        accuracy_mean = np.mean(accuracy_list)
        best_index = np.argmax(accuracy_list)
        best_solution = solution_list[best_index]
        best_accuracy = accuracy_list[best_index]

        print(
            f"{algorithm_name} run {run_times} times , mean accuracy : {accuracy_mean*100:.2f}%, best solution: {best_solution}, best accuracy : {best_accuracy*100:.2f}%"
        )
        print(f"{algorithm_name} 对{self.Dataset}数据集的特征选择，运行结束...\n")

        # 将结果保存到文件
        if is_save:
            save_result(
                algorithm_name,
                accuracy_mean,
                best_solution,
                best_accuracy,
                run_times,
                self.Dataset,
            )

        if is_plot:
            save_figure(algorithm_name, f_list, run_times, self.Dataset)

        return accuracy_mean, best_solution, best_accuracy, f_list

    def compare(
        self,
        algorithm_list,  # 算法名称列表
        run_times=1,  # 运行次数
    ):
        accuracy_list = []

        for i in range(len(algorithm_list)):
            mean_accuracy, _, _, f_list = self.fit(
                algorithm_list[i], run_times, False, False
            )
            accuracy_list.append(mean_accuracy)
            median = np.median(f_list, axis=0)
            sample_interval = 20
            median = median[::sample_interval]
            plt.plot(median, label=algorithm_list[i])

        # 根据accuracy_list平均准确率，并找到最优解
        sorted_index = np.argsort(accuracy_list)[::-1]
        print(f"\n对{self.Dataset}数据集的特征选择，按平均准确率降序排序,比较结果如下:")
        for i in range(len(algorithm_list)):
            print(
                f"{algorithm_list[sorted_index[i]]}: {accuracy_list[sorted_index[i]]*100:.2f}%"
            )

        plt.title(f"{self.Dataset} Comparison")
        plt.legend()
        plt.xlabel("iteration")
        plt.ylabel("fitness")
        plt.savefig(f"./output/{self.Dataset}_Comparison.png")

        return accuracy_list

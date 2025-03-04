from module.utils import *
from module.BPSO import *
from module.BPSO_Obl import *
from module.DE import *
from module.DE_JADE import *
from module.DE_SHADE import *
from module.DE_LSHADE import *
from module.DE_RL_LSHADE import *
from matplotlib import pyplot as plt


class FeatureSelect:
    def __init__(self, X, y, M=20):
        """
        初始化特征选择对象

        参数:
        X: 特征矩阵，形状为 (样本数量, 特征数量)
        y: 目标类别标签，形状为 (样本数量,)
        M: 特征选择次数，默认值为1
        """
        self.X = X
        self.y = y
        self.M = M

    def fit_BPSO(self, name):
        bpso = BPSO(self.X, self.y)  # 初始化BPSO对象
        bpso.fit()  # 运行BPSO算法

        print(f"\nBPSO 最优解: {bpso.global_best}, 适应度函数值: {bpso.global_best_fitness:.6f}")

        # 画图,横坐标为迭代次数，纵坐标为适应度函数值
        plt.plot(bpso.f_best, label="BPSO")
        plt.xlabel("iteration")
        plt.ylabel("fitness")
        plt.legend()
        plt.title(name)
        plt.savefig("./output/BPSO.png")
        plt.show()

    def fit_BPSO_Obl(self, name):
        bpso_obl = BPSO_Obl(self.X, self.y)  # 初始化BPSO_Obl对象
        bpso_obl.fit()  # 运行BPSO_Obl算法

        print(
            f"\nBPSO_Obl 最优解: {bpso_obl.global_best}, 适应度函数值: {bpso_obl.global_best_fitness:.6f}"
        )

        # 画图,横坐标为迭代次数，纵坐标为适应度函数值
        plt.plot(bpso_obl.f_best, label="BPSO_Obl")
        plt.xlabel("iteration")
        plt.ylabel("fitness")
        plt.legend()
        plt.title(name)
        plt.savefig("./output/BPSO_Obl.png")
        plt.show()

    def fit_DE(self, name):
        de_model = DE(self.X, self.y)  # 初始化DE对象
        de_model.fit()  # 运行DE算法

        print(
            f"\nDE 最优解: {de_model.global_best}, 适应度函数值: {de_model.global_best_fitness:.6f}"
        )

        plt.plot(de_model.f_best, label="DE")
        plt.xlabel("iteration")
        plt.ylabel("fitness")
        plt.legend()
        plt.title(name)
        plt.savefig("./output/DE.png")
        plt.show()

    def fit_DE_JADE(self, name):
        de_jade = DE_JADE(self.X, self.y)  # 初始化DE_JADE对象
        de_jade.fit()  # 运行DE_JADE算法

        print(
            f"\nDE_JADE 最优解: {de_jade.global_best}, 适应度函数值: {de_jade.global_best_fitness:.6f}"
        )

        plt.plot(de_jade.f_best, label="DE_JADE")
        plt.xlabel("iteration")
        plt.ylabel("fitness")
        plt.legend()
        plt.title(name)
        plt.savefig("./output/DE_JADE.png")
        plt.show()

    def fit_DE_SHADE(self, name):
        de_shade = DE_SHADE(self.X, self.y)
        de_shade.fit()

        print(
            f"\nDE_SHADE 最优解: {de_shade.global_best}, 适应度函数值: {de_shade.global_best_fitness:.6f}"
        )

        plt.plot(de_shade.f_best, label="DE_SHADE")
        plt.xlabel("iteration")
        plt.ylabel("fitness")
        plt.legend()
        plt.title(name)
        plt.savefig("./output/DE_SHADE.png")
        plt.show()

    def fit_DE_LSHADE(self, name):
        de_lshade = DE_LSHADE(self.X, self.y)
        de_lshade.fit()

        print(
            f"\nDE_LSHADE 最优解: {de_lshade.global_best}, 适应度函数值: {de_lshade.global_best_fitness:.6f}"
        )

        plt.plot(de_lshade.f_best, label="DE_LSHADE")
        plt.xlabel("iteration")
        plt.ylabel("fitness")
        plt.legend()
        plt.title(name)
        plt.savefig("./output/DE_LSHADE.png")
        plt.show()

    def fit_DE_RL_LSHADE(self, name):
        de_rl_lshade = DE_RL_LSHADE(self.X, self.y)
        de_rl_lshade.fit()

        print(
            f"\nDE_RL_LSHADE 最优解: {de_rl_lshade.global_best}, 适应度函数值: {de_rl_lshade.global_best_fitness:.6f}"
        )

        plt.plot(de_rl_lshade.f_best, label="DE_RL_LSHADE")
        plt.xlabel("iteration")
        plt.ylabel("fitness")
        plt.legend()
        plt.title(name)
        plt.savefig("./output/DE_RL_LSHADE.png")
        plt.show()

    def compare(self):
        bpso = BPSO(self.X, self.y)  # 初始化BPSO对象
        bpso_obl = BPSO_Obl(self.X, self.y)  # 初始化BPSO_Obl对象
        de = DE(self.X, self.y)  # 初始化DE对象
        de_jade = DE_JADE(self.X, self.y)  # 初始化DE_JADE对象

        bpso_gbest = []
        bpso_obl_gbest = []
        de_gbest = []
        de_jade_gbest = []

        bpso_best_fitness = []
        bpso_obl_best_fitness = []
        de_best_fitness = []
        de_jade_best_fitness = []

        for i in range(self.M):
            print(f"\n第{i+1}次特征选择:")
            bpso_gbest.append(bpso.fit())
            bpso_best_fitness.append(bpso.global_best_fitness)

            bpso_obl_gbest.append(bpso_obl.fit())
            bpso_obl_best_fitness.append(bpso_obl.global_best_fitness)

            de_gbest.append(de.fit())
            de_best_fitness.append(de.global_best_fitness)

            de_jade_gbest.append(de_jade.fit())
            de_jade_best_fitness.append(de_jade.global_best_fitness)

            print(
                f"BPSO 本次最优解: \t{bpso_gbest[-1]}, 适应度函数值: {bpso_best_fitness[-1]:.6f}"
            )
            print(
                f"BPSO_Obl 本次最优解: \t{bpso_obl_gbest[-1]}, 适应度函数值: {bpso_obl_best_fitness[-1]:.6f}"
            )
            print(f"DE 本次最优解: \t\t{de_gbest[-1]}, 适应度函数值: {de_best_fitness[-1]:.6f}")
            print(
                f"DE_JADE 本次最优解: \t{de_jade_gbest[-1]}, 适应度函数值: {de_jade.global_best_fitness:.6f}"
            )

        print(f"\n运行{self.M}次特征选择的结果:")
        print(
            f"BPSO 平均适应度函数值:\t{np.mean(bpso_best_fitness):.6f} , 最优适应度函数值:{np.min(bpso_best_fitness):.6f}, 最优解:{bpso_gbest[np.argmin(bpso_best_fitness)]}"
        )
        print(
            f"BPSO_Obl 平均适应度函数值:\t{np.mean(bpso_obl_best_fitness):.6f} , 最优适应度函数值:{np.min(bpso_obl_best_fitness):.6f}, 最优解:{bpso_obl_gbest[np.argmin(bpso_obl_best_fitness)]}"
        )
        print(
            f"DE 平均适应度函数值:\t{np.mean(de_best_fitness):.6f} , 最优适应度函数值:{np.min(de_best_fitness):.6f}, 最优解:{de_gbest[np.argmin(de_best_fitness)]}"
        )
        print(
            f"DE_JADE 平均适应度函数值:\t{np.mean(de_jade_best_fitness):.6f} , 最优适应度函数值:{np.min(de_jade_best_fitness):.6f}, 最优解:{de_jade_gbest[np.argmin(de_jade_best_fitness)]}"
        )

        plt.plot(bpso_best_fitness, label="BPSO")
        plt.plot(bpso_obl_best_fitness, label="BPSO_Obl")
        plt.plot(de_best_fitness, label="DE")
        plt.plot(de_jade_best_fitness, label="DE_JADE")
        plt.xlabel("M")
        plt.ylabel("fitness")
        plt.legend()
        plt.savefig("./compare.png")
        plt.show()

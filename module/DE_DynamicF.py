from sklearn.feature_selection import mutual_info_classif
from module.utils import *
from module.Base import Base


class DE_DynamicF(Base):
    def __init__(
        self,
        X,
        y,
    ):
        super().__init__(X, y, algorithm="DE_DynamicF")

    def init_solution(self):
        super().init_solution()
        # 新增特征权重计算
        self.feature_weights = mutual_info_classif(self.X, self.y)
        # 归一化处理
        self.feature_weights = (self.feature_weights - self.feature_weights.min()) / (
            self.feature_weights.max() - self.feature_weights.min() + 1e-8
        )
        # 初始化动态F值矩阵
        self.F_matrix = np.zeros((self.size, self.dimension))
        self.update_F_matrix()

    def update_F_matrix(self):
        """根据特征权重动态更新F值矩阵"""
        for d in range(self.dimension):
            self.F_matrix[:, d] = np.random.normal(
                    self.feature_weights[d], 0.2, self.size
                )
            # 将F值限制在0到2之间
            self.F_matrix[:, d] = np.clip(self.F_matrix[:, d], 0, 2)



    def F_rand_1(self, i):
        # 从种群中随机选择三个不同的个体
        x_set = set()
        x_set.add(i)
        r = np.zeros(3, dtype=int)
        for j in range(3):
            r[j] = np.random.choice(self.size, 1)[0]
            while r[j] in x_set:
                r[j] = np.random.choice(self.size, 1)[0]
            x_set.add(r[j])

        V = self.x[r[0]] + self.F_matrix[i] * (self.x[r[1]] - self.x[r[2]])
        V = np.clip(V, 0, 1)
        return V

    def update(self, i):
        V = self.F_rand_1(i)
        U = self.x[i].copy()

        # 维度级交叉操作
        for d in range(self.dimension):
            if np.random.rand() < self.CR:
                U[d] = V[d]

        population_U = (U > 0.5).astype(int)
        f_u = fitness(
            self.X,
            self.y,
            population_U,
        )

        if f_u < self.fitness_x[i]:
            self.x[i] = U
            self.fitness_x[i] = f_u
            self.P[i] = population_U
            if f_u < self.global_best_fitness:
                self.global_best = population_U
                self.global_best_fitness = f_u

    def update_parameter(self):
        if self.FES % (5 * self.size) == 0:
            self.update_F_matrix()
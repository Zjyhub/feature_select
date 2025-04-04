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
            if self.feature_weights[d] > 0.7:
                self.F_matrix[:, d] = 0.5
            elif 0.3 <= self.feature_weights[d] <= 0.7:
                self.F_matrix[:, d] = 1.2
            else:
                self.F_matrix[:, d] = 0.0

    def similarity_selection(self, current_index):
        """相似性导向的基向量选择"""
        current_vector = self.P[current_index]
        similarities = []

        # 计算相似性得分
        for i in range(self.size):
            if i == current_index:
                continue
            intersection = np.sum(current_vector & self.P[i])
            union = np.sum(current_vector | self.P[i])
            similarities.append(intersection / (union + 1e-8))

        # 轮盘赌选择
        probabilities = np.array(similarities) / (np.sum(similarities) + 1e-8)
        return np.random.choice(
            [i for i in range(self.size) if i != current_index], p=probabilities
        )

    def dynamic_mutation(self, i):
        """动态F值变异策略"""
        # 相似性选择基向量
        base_index = self.similarity_selection(i)
        x_base = self.x[base_index]

        # 随机选择两个不同个体
        r1, r2 = np.random.choice(
            [j for j in range(self.size) if j not in [i, base_index]], 2, replace=False
        )

        # 应用动态F值
        delta = self.F_matrix[i] * (self.x[r1] - self.x[r2])
        V = x_base + delta
        return np.clip(V, 0, 1)

    def update(self, i):
        V = self.dynamic_mutation(i)
        U = self.x[i].copy()

        # 维度级交叉操作
        for d in range(self.dimension):
            # 对于F=0的维度直接保留原值
            if self.F_matrix[i, d] == 0:
                U[d] = self.x[i, d]
            elif np.random.rand() < self.CR:
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
        # 每10代更新一次F值矩阵
        if self.FES % (10 * self.size) == 0:
            self.update_F_matrix()
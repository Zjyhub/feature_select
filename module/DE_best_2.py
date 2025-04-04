from module.utils import *
from module.Base import Base


class DE_best_2(Base):
    def __init__(
        self,
        X,
        y,
    ):
        super().__init__(X, y, algorithm="DE_best_2")

    # 更新种群
    def update(self, i):
        # 选择不同变异策略
        V = self.F_best_2(i)

        # 交叉操作，根据交叉概率CR生成新的个体U
        U = self.x[i].copy()
        for j in range(self.dimension):
            if np.random.rand() < self.CR:
                U[j] = V[j]
        population_U = (U > 0.5).astype(int)

        # 选择操作，选择适应度函数值更小的个体
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

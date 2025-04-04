from scipy.stats import cauchy
from module.utils import *
from module.Base import Base


class DE_JADE(Base):
    def __init__(
        self,
        X,
        y,
        u_F=0.5,
        u_CR=0.5,
        c=0.2,
        p=0.05,
    ):
        """
        初始化DE算法对象

        参数:
        u_F: 根据柯西分布生成缩放因子F的参数，默认值为0.5
        u_CR: 根据正态分布生成交叉概率CR的参数，默认值为0.5
        c: 控制参数，用来更新u_F和u_CR，默认值为0.2
        p: 控制参数，用来选择前p%的个体，默认值为0.05
        """
        super().__init__(X, y, algorithm="DE_JADE")
        self.u_F = u_F
        self.u_CR = u_CR
        self.c = c
        self.p = p

    # 初始化种群
    def init_solution(self):
        super().init_solution()
        self.A = np.zeros((0, self.dimension))  # 存储被淘汰的父代个体
        self.S_F = np.zeros(0)  # 存储成功替换父代的缩放因子
        self.S_CR = np.zeros(0)  # 存储成功替换父代的交叉概率

    # 计算Lehmer均值
    def mean_lehmer(self, p=2):
        numerator = np.sum(np.power(self.S_F, p))
        denominator = np.sum(np.power(self.S_F, p - 1))
        # 防止分母为0
        if denominator == 0:
            return 0
        return numerator / denominator

    # 从前p%的个体中随机选择一个个体
    def get_random_from_top(self):
        k = int(self.size * self.p)
        partitioned = np.partition(self.fitness_x, k)
        threshold = partitioned[k]
        indices = np.where(self.fitness_x <= threshold)[0]
        return self.x[np.random.choice(indices, 1)[0]]

    # 变异策略
    def F_current_to_pbest(self, i):
        # 在前p%的个体中随机选择一个个体
        pbest = self.get_random_from_top()
        r1 = np.random.choice(self.size, 1)[0]
        for j in range(self.size):
            if r1 != i:
                break
            r1 = np.random.choice(self.size, 1)[0]
        x_r1 = self.x[r1]

        A_x = np.concatenate((self.x, self.A))
        x_r2 = A_x[np.random.choice(len(A_x), 1)[0]]

        # 生成新的个体
        V = self.x[i] + self.F * (pbest - self.x[i]) + self.F * (x_r1 - x_r2)
        # 检查是否越界
        V = np.clip(V, 0, 1)
        return V

    # 更新种群
    def update(self, i):
        # 初始化缩放因子F和交叉概率CR
        self.F = cauchy.rvs(loc=0.5, scale=0.1, size=1)[0]  # 从柯西分布中生成F
        self.CR = np.random.normal(self.u_CR, 0.1, 1)[0]  # 从正态分布中生成CR
        # 将CR限制在[0,1]之间
        self.F = np.clip(self.F, 0, 1)
        self.CR = np.clip(self.CR, 0, 1)

        # 变异操作，根据变异策略生成新的个体V
        V = self.F_current_to_pbest(i)

        # 交叉操作，根据交叉概率CR生成新的个体U
        U = self.x[i].copy()
        j_rand = np.random.randint(0, self.dimension)
        for j in range(self.dimension):
            if np.random.rand() < self.CR or j == j_rand:
                U[j] = V[j]
        population_U = (U > 0.5).astype(int)

        # 选择操作，选择适应度函数值更小的个体
        f_u = fitness(
            self.X,
            self.y,
            population_U,
        )

        # 如果适应度函数值更小，则替换父代
        if f_u < self.fitness_x[i]:
            self.A = np.append(self.A, [self.x[i]], axis=0)
            self.P[i] = population_U
            self.x[i] = U
            self.fitness_x[i] = f_u
            self.S_F = np.append(self.S_F, self.F)
            self.S_CR = np.append(self.S_CR, self.CR)

            if len(self.A) > self.size:
                self.A = np.delete(self.A, np.random.randint(0, len(self.A), 1), axis=0)

            if f_u < self.global_best_fitness:
                self.global_best = population_U
                self.global_best_fitness = f_u

    def update_parameter(self):
        # 更新缩放因子F和交叉概率CR
        if len(self.S_CR) > 0:
            self.u_CR = (1 - self.c) * self.u_CR + self.c * np.mean(
                self.S_CR
            )  # 如果c不为0，则利用成功替换父代的CR的均值来更新u_CR
        else:
            self.u_CR = (1 - self.c) * self.u_CR + self.c * self.CR
        self.u_F = (
            1 - self.c
        ) * self.u_F + self.c * self.mean_lehmer()  # 如果c不为0，则利用成功替换父代的F的lehmer均值来更新u_F

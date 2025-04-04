from scipy.stats import cauchy
from module.utils import *
from module.Base import Base


class DE_SHADE(Base):
    def __init__(
        self,
        X,
        y,
        u_F=0.5,
        u_CR=0.5,
        p=0.1,
        H=5,
        r_arc=2,
        mcr_terminal=0.6,
    ):
        """
        初始化DE算法对象

        参数:
        u_F: 根据柯西分布生成缩放因子F的参数，默认值为0.5
        u_CR: 根据正态分布生成交叉概率CR的参数，默认值为0.5
        p: 控制参数，用来选择前p%的个体，默认值为0.05
        H: 控制参数，表示记录的M_F和M_CR的长度，默认值为5
        r_arc: 控制参数，控制被淘汰的父代个体的数量为r_arc*size，默认值为2
        mcr_terminal: 控制参数，当M_CR小于mcr_terminal时停止，默认值为0.6
        """
        super().__init__(X, y, algorithm="DE_SHADE")
        self.u_F = u_F
        self.u_CR = u_CR
        self.p = p
        self.H = H
        self.r_arc = r_arc
        self.mcr_terminal = mcr_terminal

    # 初始化种群
    def init_solution(self):
        """
        参数:
        A: 存储被淘汰的父代个体，形状为 (被淘汰个体数量, 特征数量)
        S_F: 存储成功替换父代的缩放因子，形状为 (成功替换个体数量,)
        S_CR: 存储成功替换父代的交叉概率，形状为 (成功替换个体数量,)
        M_F: 存储最近H次迭代的缩放因子，形状为 (H,)
        M_CR: 存储最近H次迭代的交叉概率，形状为 (H,)
        """
        super().init_solution()
        self.F = -1
        self.CR = -1
        self.A = np.zeros((0, self.dimension))
        self.S_F = np.zeros(0)
        self.S_CR = np.zeros(0)
        self.M_F = np.full(self.H, self.u_F)
        self.M_CR = np.full(self.H, self.u_CR)

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
        threshold = partitioned[k]  # 找到第k个最小值
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
        # 在[0,H)之间随机选择一个整数]
        r_i = np.random.choice(self.H, 1)[0]

        # 初始化缩放因子F和交叉概率CR
        self.F = cauchy.rvs(loc=self.M_F[r_i], scale=0.1, size=1)[
            0
        ]  # 从柯西分布中生成F
        if self.M_CR[r_i] == self.mcr_terminal:
            self.CR = 0
        else:
            self.CR = np.random.normal(self.M_CR[r_i], 0.1, 1)[0]  # 从正态分布中生成CR
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

        if f_u < self.fitness_x[i]:
            self.P[i] = population_U
            self.x[i] = U
            self.fitness_x[i] = f_u
            self.S_F = np.append(self.S_F, self.F)
            self.S_CR = np.append(self.S_CR, self.CR)

            self.A = np.append(self.A, [self.x[i]], axis=0)
            if len(self.A) > self.size * self.r_arc:
                self.A = np.delete(self.A, np.random.randint(0, len(self.A), 1), axis=0)

            if f_u < self.global_best_fitness:
                self.global_best = population_U
                self.global_best_fitness = f_u

    def update_parameter(self):
        # 更新M_F和M_CR
        if len(self.S_F) > 0 and len(self.S_CR) > 0:
            for i in range(self.H):
                if self.M_CR[i] == self.mcr_terminal:
                    continue
                else:
                    self.M_CR[i] = np.mean(self.S_CR)
                self.M_F[i] = self.mean_lehmer()

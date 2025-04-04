from module.utils import *
from module.Base import Base


class BPSO(Base):
    def __init__(
        self,
        X,
        y,
        v_high=6,
        c1=2.0,
        c2=2.0,
        r=1,
        w=1,
        w_max=0.9,
        w_min=0.4,
    ):
        """
        初始化BPSO算法对象

        参数:
        v_high: 粒子速度最大值，速度范围为 [-v_high, v_high]，默认值为6
        c1: 加速度系数，用于控制个体最优的影响，默认值为2.0
        c2: 加速度系数，用于控制全局最优的影响，默认值为2.0
        r: 选择函数(r为1则使用sigmoid函数，否则使用tanh函数)，默认值为1
        w: 惯性权重，用于控制上一次速度的影响，默认值为1
        w_max: 惯性权重最大值，默认值为0.9
        w_min: 惯性权重最小值，默认值为0.4
        """
        super().__init__(X, y, algorithm="BPSO")
        self.v_high = v_high
        self.c1 = c1
        self.c2 = c2
        self.r = r
        self.w = w
        self.w_max = w_max
        self.w_min = w_min

    # 初始化粒子群
    def init_solution(self):
        """
        参数:
        x: 粒子群位置，形状为 (粒子数量, 特征数量)
        p_best: 粒子群个体最优位置，形状为 (粒子数量, 特征数量)
        v: 粒子群速度，形状为 (粒子数量, 特征数量)
        """
        super().init_solution()
        self.x = np.random.randint(0, 2, size=(self.size, self.dimension), dtype=int)
        self.p_best = self.x.copy()
        self.v = np.zeros((self.size, self.dimension))
        # 更新惯性权重
        self.w = self.w_max - (self.w_max - self.w_min) * self.FES / self.max_FES

    # 粒子群更新
    def update(self, i):

        # 更新当前粒子的速度
        self.v[i] = (
            self.w * self.v[i]
            + self.c1 * np.random.rand() * (self.p_best[i] - self.x[i])
            + self.c2 * np.random.rand() * (self.global_best - self.x[i])
        )
        self.v[i] = np.clip(self.v[i], -self.v_high, self.v_high)  # 限制速度范围

        # 更新位置,遍历每个维度
        for j in range(self.dimension):
            # 如果r为1则使用sigmoid函数，否则使用tanh函数
            if self.r == 1:
                num = sigmoid(self.v[i][j])
            else:
                num = tanh(self.v[i][j])

            # 如果随机数大于x[i][j]，则x[i][j]取1，否则取0
            if num > np.random.rand():
                self.x[i][j] = 1
            else:
                self.x[i][j] = 0

        self.x[i] = np.clip(self.x[i], 0, 1)  # 限制位置范围
        f_new = fitness(
            self.X,
            self.y,
            self.x[i],
        )  # 计算当前粒子的适应度函数值

        # 更新个体最优位置
        if f_new < self.fitness_x[i]:
            self.p_best[i] = self.x[i]
            self.fitness_x[i] = f_new
            # 更新全局最优位置
            if f_new < self.global_best_fitness:
                self.global_best = self.p_best[i]
                self.global_best_fitness = f_new

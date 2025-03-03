from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import cauchy
from module.utils import *


class DE_RL_LSHADE:
    def __init__(
        self,
        X,
        y,
        iterations=100,
        init_size=20,
        min_size=10,
        alpha=0.99,
        beta=0.01,
        u_F=0.5,
        u_CR=0.5,
        p=0.1,
        H=5,
        r_arc=2.0,
        mcr_terminal=0.6,
        max_FES=1000,
    ):
        """
        初始化DE算法对象

        参数:
        X: 特征矩阵，形状为 (样本数量, 特征数量)
        y: 目标类别标签，形状为 (样本数量,)
        iterations: 迭代次数，默认值为100
        init_size: 初始种群数量，默认值为20
        min_size: 最小种群数量，默认值为10
        u_F: 根据柯西分布生成缩放因子F的参数，默认值为0.5
        u_CR: 根据正态分布生成交叉概率CR的参数，默认值为0.5
        c: 控制参数，用来更新u_F和u_CR，默认值为0.2
        p: 控制参数，用来选择前p%的个体，默认值为0.05
        H: 控制参数，表示记录的M_F和M_CR的长度，默认值为5
        r_arc: 控制参数，控制被淘汰的父代个体的数量为r_arc*size，默认值为2
        cr_terminal: 控制参数，当M_CR小于mcr_terminal时停止，默认值为0.6
        max_FES: 最大评估次数，默认值为1000
        """
        self.X = X
        self.y = y
        self.iterations = iterations
        self.init_size = init_size
        self.min_size = min_size
        self.alpha = alpha
        self.beta = beta
        self.u_F = u_F
        self.u_CR = u_CR
        self.p = p
        self.H = H
        self.r_arc = r_arc
        self.mcr_terminal = mcr_terminal

        self.max_FES = max_FES
        self.dimension = X.shape[1]

        self.F = -1
        self.CR = -1
        self.size = init_size
        self.P = np.zeros((self.size, self.dimension)).astype(int)  # 种群
        self.x = np.zeros((self.size, self.dimension))
        self.pbest = np.zeros((self.size, self.dimension))  # 个体历史最优位置
        self.pbest_fitness = np.zeros(self.size)  # 个体历史最优适应度
        self.A = np.zeros((0, self.dimension))  # 存储被淘汰的父代个体
        self.S_F = np.zeros(0)  # 存储成功替换父代的缩放因子
        self.S_CR = np.zeros(0)  # 存储成功替换父代的交叉概率
        self.M_F = np.full(self.H, self.u_F)  # 存储最近H次迭代的缩放因子
        self.M_CR = np.full(self.H, self.u_CR)  # 存储最近H次迭代的交叉概率
        self.FES = 0  # 评估次数
        self.global_best_fitness = float("inf")  # 全局最优适应度
        self.global_best = np.zeros(self.dimension).astype(int)  # 全局最优解
        self.f_best = []
        self.Q_table = np.zeros((2, 6))
        self.knn = KNeighborsClassifier(n_neighbors=5)
        pass

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
        partitioned = np.partition(self.pbest_fitness, k)
        threshold = partitioned[k]  # 找到第k个最小值
        indices = np.where(self.pbest_fitness <= threshold)[0]
        return self.pbest[np.random.choice(indices, 1)[0]]

    # 变异策略
    # DE/rand/1
    def F_rand_1(self, i):
        # 从种群中随机选择三个不同的个体
        x_set = set()
        x_set.add(i)
        r = np.zeros(3)
        for j in range(3):
            r[j] = np.random.choice(self.size, 1)[0]
            while r[j] in x_set:
                r[j] = np.random.choice(self.size, 1)[0]
            x_set.add(r[j])

        V = self.x[r[0]] + self.F * (self.x[r[1]] - self.x[r[2]])
        V = np.clip(V, 0, 1)
        return V

    # DE/rand/2
    def F_rand_2(self, i):
        # 从种群中随机选择五个不同
        x_set = set()
        x_set.add(i)
        r = np.zeros(5)
        for j in range(5):
            r[j] = np.random.choice(self.size, 1)[0]
            while r[j] in x_set:
                r[j] = np.random.choice(self.size, 1)[0]
            x_set.add(r[j])

        V = (
            self.x[r[0]]
            + self.F * (self.x[r[1]] - self.x[r[2]])
            + self.F * (self.x[r[3]] - self.x[r[4]])
        )
        V = np.clip(V, 0, 1)
        return V

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

    def reduce_population(self):
        # 根据适应度函数值对种群进行排序
        sorted_index = np.argsort(self.pbest_fitness)
        # 计算被淘汰的父代个体数量
        num_to_remove = int((self.min_size - self.size) * self.FES / self.max_FES)
        # 从种群中删除适应度函数值最大的个体
        if num_to_remove <= -1:
            self.P = np.delete(self.P, sorted_index[num_to_remove:], axis=0)
            self.x = np.delete(self.x, sorted_index[num_to_remove:], axis=0)
            self.pbest = np.delete(self.pbest, sorted_index[num_to_remove:], axis=0)
            self.pbest_fitness = np.delete(
                self.pbest_fitness, sorted_index[num_to_remove:]
            )
            self.size += num_to_remove
            len_A = int(self.size * self.r_arc)
            if len(self.A) > len_A:
                self.A = np.delete(
                    self.A,
                    np.random.randint(0, len(self.A), len(self.A) - len_A),
                    axis=0,
                )

    # 初始化种群
    def init_solution(self):
        self.P = np.zeros((self.size, self.dimension)).astype(int)  # 种群
        self.x = np.zeros((self.size, self.dimension))
        self.pbest = np.zeros((self.size, self.dimension))  # 个体历史最优位置
        self.A = np.zeros((0, self.dimension))  # 存储被淘汰的父代个体
        self.S_F = np.zeros(0)  # 存储成功的缩放因子
        self.S_CR = np.zeros(0)  # 存储成功的交叉概率
        self.M_F = np.full(self.H, self.u_F)  # 存储最近H次迭代的缩放因子
        self.M_CR = np.full(self.H, self.u_CR)  # 存储最近H次迭代的交叉概率
        self.FES = 0  # 评估次数
        self.global_best_fitness = float("inf")  # 全局最优适应度
        self.global_best = np.zeros(self.dimension).astype(int)  # 全局最优解
        self.f_best = []
        for i in range(self.size):
            # 将x[i]初始化为0-1之间的随机数
            self.x[i] = np.random.rand(self.dimension)
            # 根据x[i]每个特征的值是否大于0.5来决定P[i]的值是否为1
            self.P[i] = (self.x[i] > 0.5).astype(int)
            f_new = fitness(
                self.alpha,
                self.beta,
                self.dimension,
                self.X,
                self.y,
                self.P[i],
                self.knn,
            )

            # 更新个体历史最优位置和全局最优位置
            self.pbest[i] = self.x[i]
            self.pbest_fitness[i] = f_new
            if f_new < self.global_best_fitness:
                self.global_best = self.P[i]
                self.global_best_fitness = f_new

    # 更新种群
    def update(self):
        for t in range(self.iterations):
            if t % 10 == 0:
                print(
                    f"当前最优解x: {self.global_best}, fitness: {self.global_best_fitness:.6f}"
                )
            for i in range(self.size):
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
                    self.alpha,
                    self.beta,
                    self.dimension,
                    self.X,
                    self.y,
                    population_U,
                    self.knn,
                )
                f_x = fitness(
                    self.alpha,
                    self.beta,
                    self.dimension,
                    self.X,
                    self.y,
                    self.P[i],
                    self.knn,
                )
                if f_u < f_x:
                    self.P[i] = population_U
                    self.x[i] = U
                    self.S_F = np.append(self.S_F, self.F)
                    self.S_CR = np.append(self.S_CR, self.CR)

                    self.A = np.append(self.A, [self.x[i]], axis=0)
                    if len(self.A) > int(self.size * self.r_arc):
                        self.A = np.delete(
                            self.A, np.random.randint(0, len(self.A), 1), axis=0
                        )
                    if f_u < self.pbest_fitness[i]:
                        self.pbest[i] = self.x[i]
                        self.pbest_fitness[i] = f_u
                    if f_u < self.global_best_fitness:
                        self.global_best = population_U
                        self.global_best_fitness = f_u

                self.FES += 1
                if self.FES >= self.max_FES:
                    self.f_best.append(self.global_best_fitness)
                    return
            # 更新M_F和M_CR
            if len(self.S_F) > 0 and len(self.S_CR) > 0:
                for i in range(self.H):
                    if self.M_CR[i] == self.mcr_terminal:
                        continue
                    else:
                        self.M_CR[i] = np.mean(self.S_CR)
                    self.M_F[i] = self.mean_lehmer()

            self.f_best.append(self.global_best_fitness)

    def fit(self):
        self.init_solution()
        self.update()
        return self.global_best

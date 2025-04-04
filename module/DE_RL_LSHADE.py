from scipy.stats import cauchy
from module.utils import *
from module.Base import Base


class DE_RL_LSHADE(Base):
    def __init__(
        self,
        X,
        y,
        init_size=global_params["size"],
        min_size=global_params["min_size"],
        u_F=0.5,
        u_CR=0.5,
        p=0.1,
        H=5,
        r_arc=2.0,
        mcr_terminal=0.6,
        gamma=0.9,
        alpha_lr=0.1,
    ):
        """
        初始化DE算法对象

        参数:
        min_size: 最小种群数量，默认值为10
        u_F: 根据柯西分布生成缩放因子F的参数，默认值为0.5
        u_CR: 根据正态分布生成交叉概率CR的参数，默认值为0.5
        c: 控制参数，用来更新u_F和u_CR，默认值为0.2
        p: 控制参数，用来选择前p%的个体，默认值为0.05
        H: 控制参数，表示记录的M_F和M_CR的长度，默认值为5
        r_arc: 控制参数，控制被淘汰的父代个体的数量为r_arc*size，默认值为2
        cr_terminal: 控制参数，当M_CR小于mcr_terminal时停止，默认值为0.6
        gamma: 控制参数，用来更新Q表的参数，默认值为0.9
        alpha_lr: 控制参数，用来更新Q表的参数，默认值为0.1
        state_num: 状态数，0表示当前个体优于之前的父代，1表示当前个体劣于之前的父代
        strategies: 策略数
        """
        super().__init__(X, y, algorithm="DE_RL_LSHADE")

        self.init_size = init_size
        self.min_size = min_size
        self.u_F = u_F
        self.u_CR = u_CR
        self.p = p
        self.H = H
        self.r_arc = r_arc
        self.mcr_terminal = mcr_terminal
        self.gamma = gamma
        self.alpha_lr = alpha_lr

        self.state_num = 2
        self.strategies = 7

    # 初始化种群
    def init_solution(self):
        """
        参数:
        A: 存储被淘汰的父代个体，形状为 (被淘汰个体数量, 特征数量)
        S_F: 存储成功替换父代的缩放因子，形状为 (成功替换个体数量,)
        S_CR: 存储成功替换父代的交叉概率，形状为 (成功替换个体数量,)
        M_F: 存储最近H次迭代的缩放因子，形状为 (H,)
        M_CR: 存储最近H次迭代的交叉概率，形状为 (H,)
        State: 记录每个个体的状态，0表示当前个体优于之前的父代，1表示当前个体劣于之前的父代
        Q_table: Q表, 形状为 (种群大小, 状态数, 策略数)，用于存储每个个体在不同状态下选择不同策略的Q值
        """
        super().init_solution()
        self.size = self.init_size
        self.A = np.zeros((0, self.dimension))  # 存储被淘汰的父代个体
        self.S_F = np.zeros(0)  # 存储成功替换父代的缩放因子
        self.S_CR = np.zeros(0)  # 存储成功替换父代的交叉概率
        self.M_F = np.full(self.H, self.u_F)  # 存储最近H次迭代的缩放因子
        self.M_CR = np.full(self.H, self.u_CR)  # 存储最近H次迭代的交叉概率
        self.State = np.zeros(
            self.size, dtype=int
        )  # 记录每个个体的状态，0表示当前个体优于之前的父代，1表示当前个体劣于之前的父代
        self.Q_table = np.zeros(
            (self.size, self.state_num, self.strategies)
        )  # Q表, 2个状态，7个动作，每个个体有一个Q表

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

    # DE/current-to-pbest/1
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

    # 策略选择
    def strategy_choice(self, i):
        choice_prob = np.zeros(7)
        q_sum = np.sum(np.exp(self.Q_table[i][self.State[i]]))
        for j in range(7):
            choice_prob[j] = np.exp(self.Q_table[i][self.State[i], j]) / q_sum
        # 判断求和是否为1
        if np.sum(choice_prob) != 1:
            choice_prob[0] += 1 - np.sum(choice_prob)
        choice = np.random.choice(7, 1, p=choice_prob)[0]
        return choice

    # 更新Q表
    def update_Q_table(self, i, choice, isbetter):
        if isbetter:
            reward = 1
        else:
            reward = 0
        self.Q_table[i][self.State[i], choice] = self.Q_table[i][
            self.State[i], choice
        ] + self.alpha_lr * (
            reward
            + self.gamma * np.max(self.Q_table[i][1 - self.State[i]])
            - self.Q_table[i][self.State[i], choice]
        )

    def reduce_population(self):
        # 根据适应度函数值对种群进行排序
        sorted_index = np.argsort(self.fitness_x)
        # 计算被淘汰的父代个体数量
        num_to_remove = int((self.min_size - self.size) * self.FES / self.max_FES)
        # 从种群中删除适应度函数值最大的个体
        if num_to_remove <= -1:
            self.P = np.delete(self.P, sorted_index[num_to_remove:], axis=0)
            self.x = np.delete(self.x, sorted_index[num_to_remove:], axis=0)
            self.Q_table = np.delete(self.Q_table, sorted_index[num_to_remove:], axis=0)
            self.State = np.delete(self.State, sorted_index[num_to_remove:])
            self.fitness_x = np.delete(self.fitness_x, sorted_index[num_to_remove:])
            self.size += num_to_remove
            len_A = int(self.size * self.r_arc)
            if len(self.A) > len_A:
                self.A = np.delete(
                    self.A,
                    np.random.randint(0, len(self.A), len(self.A) - len_A),
                    axis=0,
                )

    # 更新种群
    def update(self, i):

        # 在[0,H)之间随机选择一个整数
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

        # 策略选择
        choice = self.strategy_choice(self.State[i])
        # 变异操作，根据变异策略生成新的个体V
        if choice == 0:
            V = self.F_rand_1(i)
        elif choice == 1:
            V = self.F_rand_2(i)
        elif choice == 2:
            V = self.F_best_1(i)
        elif choice == 3:
            V = self.F_best_2(i)
        elif choice == 4:
            V = self.F_current_to_rand_1(i)
        elif choice == 5:
            V = self.F_current_to_best_1(i)
        elif choice == 6:
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
            if len(self.A) > int(self.size * self.r_arc):
                self.A = np.delete(self.A, np.random.randint(0, len(self.A), 1), axis=0)
            if f_u < self.global_best_fitness:
                self.global_best = population_U
                self.global_best_fitness = f_u

            # 更新Q表
            self.update_Q_table(i, choice, True)
            self.State[i] = 0
        else:
            # 更新Q表
            self.update_Q_table(i, choice, False)
            self.State[i] = 1

    def update_parameter(self):
        self.reduce_population()
        # 更新M_F和M_CR
        if len(self.S_F) > 0 and len(self.S_CR) > 0:
            for i in range(self.H):
                if self.M_CR[i] == self.mcr_terminal:
                    continue
                else:
                    self.M_CR[i] = np.mean(self.S_CR)
                self.M_F[i] = self.mean_lehmer()

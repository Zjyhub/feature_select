from module.utils import *
from module.Base import Base
from sklearn.feature_selection import mutual_info_classif


class DE_RL_DynamicF(Base):
    def __init__(
        self,
        X,
        y,
        gamma=0.9,
        alpha_lr=0.1,
    ):
        """
        初始化DE算法对象

        参数:
        gamma: 折扣因子，默认值为0.9
        alpha_lr: 学习率，默认值为0.1
        state_num: 状态数，0表示当前个体优于之前的父代，1表示当前个体劣于之前的父代
        strategies: 策略数
        """
        super().__init__(X, y, algorithm="DE_RL_DynamicF")
        self.gamma = gamma
        self.alpha_lr = alpha_lr
        self.state_num = 2
        self.strategies = 6

    # 初始化种群
    def init_solution(self):
        """
        参数:
        State: 记录每个个体的状态，0表示当前个体优于之前的父代，1表示当前个体劣于之前的父代
        Q_table: Q表, 形状为 (种群大小, 状态数, 策略数)，用于存储每个个体在不同状态下选择不同策略的Q值
        """
        super().init_solution()
        self.State = np.zeros(self.size, dtype=int)
        self.Q_table = np.zeros((self.size, self.state_num, self.strategies))
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

    # 策略选择
    def strategy_choice(self, i):
        choice_prob = np.zeros(self.strategies)
        q_sum = np.sum(np.exp(self.Q_table[i][self.State[i]]))
        for j in range(self.strategies):
            choice_prob[j] = np.exp(self.Q_table[i][self.State[i], j]) / q_sum
        # 判断求和是否为1
        if np.sum(choice_prob) != 1:
            choice_prob[0] += 1 - np.sum(choice_prob)
        choice = np.random.choice(self.strategies, 1, p=choice_prob)[0]
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

# DE/rand/2
    def F_rand_2(self, i):
        # 从种群中随机选择五个不同
        x_set = set()
        x_set.add(i)
        r = np.zeros(5, dtype=int)
        for j in range(5):
            r[j] = np.random.choice(self.size, 1)[0]
            while r[j] in x_set:
                r[j] = np.random.choice(self.size, 1)[0]
            x_set.add(r[j])

        V = (
            self.x[r[0]]
            + self.F_matrix[i] * (self.x[r[1]] - self.x[r[2]])
            + self.F_matrix[i] * (self.x[r[3]] - self.x[r[4]])
        )
        V = np.clip(V, 0, 1)
        return V

    # DE/best/1
    def F_best_1(self, i):
        # 选择两个不同的个体和全局最优个体
        # 根据适应度函数值获得全局最优个体的索引
        best = np.argmin(self.fitness_x)
        x_set = set()
        x_set.add(i)
        x_set.add(best)
        r = np.zeros(2, dtype=int)
        for j in range(2):
            r[j] = np.random.choice(self.size, 1)[0]
            while r[j] in x_set:
                r[j] = np.random.choice(self.size, 1)[0]
            x_set.add(r[j])

        V = self.x[best] + self.F_matrix[i] * (self.x[r[0]] - self.x[r[1]])
        V = np.clip(V, 0, 1)
        return V

    # DE/best/2
    def F_best_2(self, i):
        # 选择四个不同的个体和全局最优个体
        best = np.argmin(self.fitness_x)
        x_set = set()
        x_set.add(i)
        x_set.add(best)
        r = np.zeros(4, dtype=int)
        for j in range(4):
            r[j] = np.random.choice(self.size, 1)[0]
            while r[j] in x_set:
                r[j] = np.random.choice(self.size, 1)[0]
            x_set.add(r[j])

        V = (
            self.x[best]
            + self.F_matrix[i] * (self.x[r[0]] - self.x[r[1]])
            + self.F_matrix[i] * (self.x[r[2]] - self.x[r[3]])
        )
        V = np.clip(V, 0, 1)
        return V

    # DE/current-to-rand/1
    def F_current_to_rand_1(self, i):
        # 选择三个不同的个体
        x_set = set()
        x_set.add(i)
        r = np.zeros(3, dtype=int)
        for j in range(3):
            r[j] = np.random.choice(self.size, 1)[0]
            while r[j] in x_set:
                r[j] = np.random.choice(self.size, 1)[0]
            x_set.add(r[j])

        V = (
            self.x[i]
            + self.F_matrix[i] * (self.x[i] - self.x[r[0]])
            + self.F_matrix[i] * (self.x[r[1]] - self.x[r[2]])
        )
        V = np.clip(V, 0, 1)
        return V

    # DE/current-to-best/1
    def F_current_to_best_1(self, i):
        # 选择两个不同的个体
        best = np.argmin(self.fitness_x)
        x_set = set()
        x_set.add(i)
        r = np.zeros(2, dtype=int)
        for j in range(2):
            r[j] = np.random.choice(self.size, 1)[0]
            while r[j] in x_set:
                r[j] = np.random.choice(self.size, 1)[0]
            x_set.add(r[j])

        V = (
            self.x[i]
            + self.F_matrix[i] * (self.x[i] - self.x[best])
            + self.F_matrix[i] * (self.x[r[0]] - self.x[r[1]])
        )
        V = np.clip(V, 0, 1)
        return V


    # 更新种群
    def update(self, i):
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
            self.x[i] = U
            self.fitness_x[i] = f_u
            self.P[i] = population_U

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
        if self.FES % (5 * self.size) == 0:
            self.update_F_matrix()
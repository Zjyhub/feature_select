from module.utils import *
from module.Base import Base


class DE_RL(Base):
    def __init__(
        self,
        X,
        y,
        size=global_params["size"],
        F=0.5,
        CR=0.5,
        gamma=0.9,
        alpha_lr=0.1,
        max_FES=global_params["max_FES"],
    ):
        """
        初始化DE算法对象

        参数:
        gamma: 折扣因子，默认值为0.9
        alpha_lr: 学习率，默认值为0.1
        state_num: 状态数，0表示当前个体优于之前的父代，1表示当前个体劣于之前的父代
        strategies: 策略数
        """
        super().__init__(X, y, size, F, CR, max_FES, algorithm="DE_RL")
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

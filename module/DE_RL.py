from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import cauchy
from module.utils import *


class DE_RL:
    def __init__(
        self,
        X,
        y,
        init_size=global_params["size"],
        min_size=10,
        alpha=global_params["alpha"],
        beta=global_params["beta"],
        F=0.5,
        CR=0.5,
        gamma=0.9,
        alpha_lr=0.1,
        max_FES=global_params["max_FES"],
    ):
        """
        初始化DE算法对象

        参数:
        X: 特征矩阵，形状为 (样本数量, 特征数量)
        y: 目标类别标签，形状为 (样本数量,)
        size: 初始种群数量，默认值为20
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
        max_FES: 最大评估次数，默认值为1000
        """
        self.X_train, self.y_train = X, y
        self.size = init_size
        self.init_size = init_size
        self.min_size = min_size
        self.alpha = alpha
        self.beta = beta
        self.F = F
        self.CR = CR
        self.gamma = gamma
        self.alpha_lr = alpha_lr
        self.max_FES = max_FES

        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.dimension = X.shape[1]
        self.knn = KNeighborsClassifier(n_neighbors=5)

    # 初始化种群
    def init_solution(self):
        self.size = self.init_size
        self.P = np.zeros((self.size, self.dimension), dtype=int)  # 种群
        self.x = np.zeros((self.size, self.dimension))
        self.fitness_x = np.zeros(self.size)  # 个体历史最优适应度
        self.FES = 0  # 评估次数
        self.global_best_fitness = float("inf")  # 全局最优适应度
        self.global_best = np.zeros(self.dimension, dtype=int)  # 全局最优解
        self.f_best = []  # 存储全局最优适应度值
        self.State = np.zeros(
            self.size, dtype=int
        )  # 记录每个个体的状态，0表示当前个体优于之前的父代，1表示当前个体劣于之前的父代
        self.Q_table = np.zeros((self.size, 2, 7))  # Q表, 2个状态，7个动作，每个个体有一个Q表
        self.t = tqdm(total=self.max_FES, desc="DE_RL_LSHADE", bar_format=bar_format)
        for i in range(self.size):
            # 将x[i]初始化为0-1之间的随机数
            self.x[i] = np.random.rand(self.dimension)
            # 根据x[i]每个特征的值是否大于0.5来决定P[i]的值是否为1
            self.P[i] = (self.x[i] > 0.5).astype(int)
            f_new = fitness(
                self.alpha,
                self.beta,
                self.dimension,
                self.X_train,
                self.y_train,
                self.P[i],
                self.knn,
            )

            # 更新个体历史最优位置和全局最优位置
            self.fitness_x[i] = f_new
            if f_new < self.global_best_fitness:
                self.global_best = self.P[i]
                self.global_best_fitness = f_new

    # 变异策略
    # DE/rand/1
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

        V = self.x[r[0]] + self.F * (self.x[r[1]] - self.x[r[2]])
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
            + self.F * (self.x[r[1]] - self.x[r[2]])
            + self.F * (self.x[r[3]] - self.x[r[4]])
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

        V = self.x[best] + self.F * (self.x[r[0]] - self.x[r[1]])
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
            + self.F * (self.x[r[0]] - self.x[r[1]])
            + self.F * (self.x[r[2]] - self.x[r[3]])
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
            + self.F * (self.x[i] - self.x[r[0]])
            + self.F * (self.x[r[1]] - self.x[r[2]])
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
            + self.F * (self.x[i] - self.x[best])
            + self.F * (self.x[r[0]] - self.x[r[1]])
        )
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


    # 更新种群
    def update(self):
        while self.FES < self.max_FES:
            self.t.set_postfix(
                {
                    "solution": self.global_best[:16],
                    "fitness": f"{self.global_best_fitness:.4f}",
                }
            )
            for i in tqdm(range(self.size), desc="种群进化中", leave=False):

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
                    self.alpha,
                    self.beta,
                    self.dimension,
                    self.X_train,
                    self.y_train,
                    population_U,
                    self.knn,
                )

                if f_u < self.fitness_x[i]:
                    self.P[i] = population_U
                    self.x[i] = U
                    self.fitness_x[i] = f_u

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

                self.FES += 1
                self.t.update(1)
                self.f_best.append(self.global_best_fitness)
                if self.FES >= self.max_FES:
                    return

    def fit(self):
        self.init_solution()
        self.update()
        # 计算准确率
        self.accuracy = cal_accuracy(
            self.X_train, self.y_train, self.global_best, self.knn
        )
        self.t.set_postfix(
            {
                "accuracy": f"{self.accuracy*100:.2f}%",
                "solution": self.global_best[:16],
                "fitness": f"{self.global_best_fitness:.4f}",
            }
        )
        self.t.close()
        return self.accuracy

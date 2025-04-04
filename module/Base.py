from module.utils import *


class Base:
    def __init__(
        self,
        X,
        y,
        size=global_params["size"],
        F=0.5,
        CR=0.5,
        max_FES=global_params["max_FES"],
        algorithm="",
    ):
        """
        初始化DE算法对象

        参数:
        X: 特征矩阵，形状为 (样本数量, 特征数量)
        y: 目标类别标签，形状为 (样本数量,)
        size: 种群大小
        F: 缩放因子，默认值为0.5
        CR: 交叉概率，默认值为0.5
        max_FES: 最大评估次数
        """
        self.X = X
        self.y = y
        self.size = size
        self.F = F
        self.CR = CR
        self.max_FES = max_FES

        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.dimension = X.shape[1]
        self.algorithm = algorithm

    # 初始化种群
    def init_solution(self):
        """
        参数:
        x: 种群矩阵，形状为 (种群大小, 特征数量)
        population: 种群，形状为 (种群大小, 特征数量)，其中1表示选择该特征，0表示不选择
        fitness_x: 种群适应度值，形状为 (种群大小,)
        global_best: 全局最优个体，形状为 (特征数量,)
        global_best_fitness: 全局最优个体适应度值
        FES: 当前评估次数
        f_best: 最佳适应度值列表，长度为max_FES
        t: 进度条对象，用于显示算法进度
        """
        self.x = np.random.rand(self.size, self.dimension)
        self.P = (self.x > 0.5).astype(int)
        self.fitness_x = np.zeros(self.size)
        self.global_best = np.zeros(self.dimension, dtype=int)
        self.global_best_fitness = float("inf")
        self.FES = 0
        self.f_best = np.zeros(self.max_FES)
        self.t = tqdm(total=self.max_FES, desc=self.algorithm, bar_format=bar_format)

    # 准备工作
    def prepare(self):
        # 计算适应度函数值
        # 初始种群适应度值
        for i in range(self.size):
            f_new = fitness(
                self.X,
                self.y,
                self.P[i],
            )
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

    # 训练
    def train(self):
        while self.FES < self.max_FES:
            self.t.set_postfix(
                {
                    "solution": self.global_best[:16],
                    "fitness": f"{self.global_best_fitness:.4f}",
                }
            )
            for i in tqdm(range(self.size), desc="种群进化中", leave=False):
                self.update(i)
                self.f_best[self.FES] = self.global_best_fitness
                self.FES += 1
                self.t.update(1)
                if self.FES >= self.max_FES:
                    return
            self.update_parameter()

    # 更新参数
    def update_parameter(self):
        pass

    # 更新个体
    def update(self, i):
        pass

    # 拟合函数
    def fit(self):
        self.init_solution()
        self.prepare()
        self.train()
        # 计算准确率
        self.accuracy = cal_accuracy(self.X, self.y, self.global_best)
        self.t.set_postfix(
            {
                "accuracy": f"{self.accuracy*100:.2f}%",
                "solution": self.global_best[:16],
                "fitness": f"{self.global_best_fitness:.4f}",
            }
        )
        self.t.close()
        return self.accuracy

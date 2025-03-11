from sklearn.neighbors import KNeighborsClassifier
from module.utils import *


class DE:
    def __init__(
        self,
        X,
        y,
        size=20,
        alpha=0.99,
        beta=0.01,
        F=0.5,
        CR=0.5,
        max_FES=1000,
    ):
        """
        初始化DE算法对象

        参数:
        X: 特征矩阵，形状为 (样本数量, 特征数量)
        y: 目标类别标签，形状为 (样本数量,)
        size: 种群大小，默认值为20
        F: 缩放因子，默认值为0.5
        CR: 交叉概率，默认值为0.5
        max_FES: 最大评估次数，默认值为1000
        """
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.F = F
        self.CR = CR
        self.max_FES = max_FES

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.dimension = X.shape[1]
        self.population = np.zeros((self.size, self.dimension)).astype(int)
        self.x = np.zeros((self.size, self.dimension))
        self.fitness_x = np.zeros(self.size)
        self.FES = 0
        self.global_best_fitness = float("inf")
        self.global_best = np.zeros(self.dimension).astype(int)
        self.f_best = []
        self.knn = KNeighborsClassifier(n_neighbors=5)

    # 初始化种群
    def init_solution(self):
        self.population = np.zeros((self.size, self.dimension),dtype=int)
        self.x = np.zeros((self.size, self.dimension))
        self.fitness_x = np.zeros(self.size)
        self.FES = 0
        self.global_best_fitness = float("inf")
        self.global_best = np.zeros(self.dimension,dtype=int)
        self.f_best = []
        self.t=tqdm(total=self.max_FES,desc="DE",bar_format=bar_format)
        for i in range(self.size):
            # 将x[i]初始化为0-1之间的随机数
            self.x[i] = np.random.rand(self.dimension)
            # 根据x[i]每个特征的值是否大于0.5来初始化种群
            self.population[i] = (self.x[i] > 0.5).astype(int)
            f_new = fitness(
                self.alpha,
                self.beta,
                self.dimension,
                self.X_train,
                self.y_train,
                self.population[i],
                self.knn,
            )
            self.fitness_x[i] = f_new
            if f_new < self.global_best_fitness:
                self.global_best = self.population[i]
                self.global_best_fitness = f_new

    # 变异策略
    # DE/rand/1
    def F_rand_1(self, i):
        # 从种群中随机选择三个不同的个体
        x_set = set()
        x_set.add(i)
        r = np.zeros(3,dtype=int)
        for j in range(3):
            r[j] = np.random.choice(self.size, 1)[0]
            while r[j] in x_set:
                r[j] = np.random.choice(self.size, 1)[0]
            x_set.add(r[j])

        V = self.x[r[0]] + self.F * (self.x[r[1]] - self.x[r[2]])
        V = np.clip(V, 0, 1)
        return V    

    # 更新种群
    def update(self):
        while self.FES < self.max_FES:
            self.t.set_postfix({"solution":self.global_best,"fitness":self.global_best_fitness})
            for i in range(self.size):
                # 选择不同变异策略
                V = self.F_rand_1(i)

                # 交叉操作，根据交叉概率CR生成新的个体U
                U = self.x[i].copy()
                for j in range(self.dimension):
                    if np.random.rand() < self.CR:
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
                    self.x[i] = U
                    self.fitness_x[i] = f_u
                    self.population[i] = population_U
                    if f_u < self.global_best_fitness:
                        self.global_best = population_U
                        self.global_best_fitness = f_u
                self.FES += 1
                self.t.update(1)
                if self.FES >= self.max_FES:
                    self.f_best.append(self.global_best_fitness)
                    return
            self.f_best.append(self.global_best_fitness)

    def fit(self):
        self.init_solution()
        self.update()

        # 使用knn算法在测试集上进行测试
        X_train = self.X_train.iloc[:,self.global_best==1]
        X_test = self.X_test.iloc[:,self.global_best==1]
        # 如果选择的特征数量为0，则返回0，否则返回在测试集上的准确率
        if X_train.shape[1] == 0:
            return 0
        self.knn.fit(X_train, self.y_train)
        y_pred = self.knn.predict(X_test)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.t.set_postfix({"solution":self.global_best,"fitness":self.global_best_fitness,"accuracy":self.accuracy})
        return self.accuracy

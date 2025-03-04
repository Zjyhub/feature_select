from sklearn.neighbors import KNeighborsClassifier
from module.utils import *


class DE:
    def __init__(
        self,
        X,
        y,
        iterations=100,
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
        iterations: 迭代次数，默认值为100
        size: 种群大小，默认值为20
        F: 缩放因子，默认值为0.5
        CR: 交叉概率，默认值为0.5
        max_FES: 最大评估次数，默认值为1000
        """
        self.X = X
        self.y = y
        self.iterations = iterations
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.F = F
        self.CR = CR
        self.max_FES = max_FES

        self.dimension = X.shape[1]
        self.population = np.zeros((self.size, self.dimension)).astype(int)
        self.x = np.zeros((self.size, self.dimension))
        self.FES = 0
        self.global_best_fitness = float("inf")
        self.global_best = np.zeros(self.dimension).astype(int)
        self.f_best = []
        self.knn = KNeighborsClassifier(n_neighbors=5)
        pass

    def F_rand(self):
        # 选择三个不同的个体的索引
        a, b, c = np.random.choice(self.size, 3, replace=False)
        # 生成新的个体
        V = self.x[a] + self.F * (self.x[b] - self.x[c])
        # 检查是否越界
        V = np.clip(V, 0, 1)
        return V

    # 初始化种群
    def init_solution(self):
        self.population = np.zeros((self.size, self.dimension)).astype(int)
        self.x = np.zeros((self.size, self.dimension))
        self.FES = 0
        self.global_best_fitness = float("inf")
        self.global_best = np.zeros(self.dimension).astype(int)
        self.f_best = []
        for i in range(self.size):
            # 将x[i]初始化为0-1之间的随机数
            self.x[i] = np.random.rand(self.dimension)
            # 根据x[i]每个特征的值是否大于0.5来初始化种群
            self.population[i] = (self.x[i] > 0.5).astype(int)
            f_new = fitness(
                self.alpha,
                self.beta,
                self.dimension,
                self.X,
                self.y,
                self.population[i],
                self.knn,
            )
            if f_new < self.global_best_fitness:
                self.global_best = self.population[i]
                self.global_best_fitness = f_new

    # 更新种群
    def update(self):
        for t in range(self.iterations):
            if t % 10 == 0:
                print(
                    f"当前最优解x: {self.global_best}, fitness: {self.global_best_fitness:.6f}"
                )
            for i in range(self.size):
                # 选择不同变异策略
                V = self.F_rand()

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
                    self.population[i],
                    self.knn,
                )
                if f_u < f_x:
                    self.population[i] = population_U
                    self.x[i] = U
                    if f_u < self.global_best_fitness:
                        self.global_best = population_U
                        self.global_best_fitness = f_u
                self.FES += 1
                if self.FES >= self.max_FES:
                    self.f_best.append(self.global_best_fitness)
                    return
            self.f_best.append(self.global_best_fitness)

    def fit(self):
        self.init_solution()
        self.update()
        return self.global_best

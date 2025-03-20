from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from module.utils import *
import numpy as np


class DE_model:
    def __init__(
        self,
        X,
        y,
        size=global_params["size"],
        alpha=global_params["alpha"],
        beta=global_params["beta"],
        F=0.5,
        CR=0.5,
        max_FES=global_params["max_FES"],
    ):
        self.X_train, self.y_train = X, y
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.F = F
        self.CR = CR
        self.max_FES = max_FES

        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.dimension = X.shape[1]
        self.population = np.zeros((self.size, self.dimension)).astype(int)
        self.x = np.zeros((self.size, self.dimension))
        self.fitness_x = np.zeros(self.size)
        self.FES = 0
        self.global_best_fitness = float("inf")
        self.global_best = np.zeros(self.dimension).astype(int)
        self.f_best = []
        self.knn = KNeighborsClassifier(n_neighbors=5)
        self.history_X = []  # 明确初始化历史记录
        self.history_f = []
        self.light_model = LinearRegression()
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        self.precise_model = GaussianProcessRegressor(kernel=kernel, alpha=1e-2)
        self.generation = 0
        self.eval_budget = max_FES

    def init_solution(self):
        self.population = np.zeros((self.size, self.dimension), dtype=int)
        self.x = np.zeros((self.size, self.dimension))
        self.fitness_x = np.zeros(self.size)
        self.FES = 0
        self.global_best_fitness = float("inf")
        self.global_best = np.zeros(self.dimension, dtype=int)
        self.f_best = []
        self.t = tqdm(total=self.max_FES, desc="DE_model", bar_format=bar_format)
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
            self.history_X.append(self.population[i])
            self.history_f.append(f_new)  # 记录初始适应度
        self.update_models()

    def update_models(self):
        """通用模型更新方法"""
        if len(self.history_X) >= 5:
            X_train = np.array(self.history_X[-100:])
            f_train = np.array(self.history_f[-100:])
            self.light_model.fit(X_train, f_train)

        if len(self.history_X) >= 10:
            X_train = np.array(self.history_X[-100:])
            f_train = np.array(self.history_f[-100:])
            self.precise_model.fit(X_train, f_train)

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

    def update(self):
        while self.FES < self.max_FES:
            self.t.set_postfix(
                {
                    "solution": self.global_best[:16],
                    "fitness": f"{self.global_best_fitness:.4f}",
                }
            )

            # 存储本代的候选解
            candidate_pool = []

            for i in tqdm(range(self.size), desc="种群进化中", leave=False):
                # ====== 原变异交叉操作 ======
                V = self.F_rand_1(i)
                U = self.x[i].copy()
                for j in range(self.dimension):
                    if np.random.rand() < self.CR:
                        U[j] = V[j]
                population_U = (U > 0.5).astype(int)
                candidate_pool.append(population_U)

            # ====== 分层筛选流程 ======
            candidates = np.array(candidate_pool)
            self.update_models()

            # 第一阶段：轻量级筛选
            if len(self.history_X) >= 5:  # 最小数据要求
                try:
                    light_scores = self.light_model.predict(candidates)
                except NotFittedError:
                    top_50 = np.arange(len(candidates))
                else:
                    top_50 = np.argsort(light_scores)[: int(self.size * 0.5)]

            # 第二阶段：精准级筛选
            if len(self.history_X) >= 10:
                precise_scores = self.precise_model.predict(candidates[top_50])
                top_k = top_50[
                    np.argsort(precise_scores)[: max(1, int(self.size * 0.1))]
                ]
            else:
                top_k = top_50[: max(1, int(self.size * 0.1))]

            # ====== 真实评估 ======
            for idx in top_k:
                if self.FES >= self.max_FES:
                    return

                candidate = candidates[idx]
                f_u = fitness(
                    self.alpha,
                    self.beta,
                    self.dimension,
                    self.X_train,
                    self.y_train,
                    candidate,
                    self.knn,
                )
                self.FES += 1
                self.t.update(1)

                # 更新历史数据
                self.history_X.append(candidate)
                self.history_f.append(f_u)

                # 更新种群
                original_idx = idx % self.size  # 对应原始个体
                if f_u < self.fitness_x[original_idx]:
                    self.x[original_idx] = U
                    self.fitness_x[original_idx] = f_u
                    self.population[original_idx] = candidate

                    if f_u < self.global_best_fitness:
                        self.global_best = candidate
                        self.global_best_fitness = f_u

                self.f_best.append(self.global_best_fitness)

            # ====== 模型更新 ======
            self.generation += 1
            if self.generation % 2 == 0:
                self.update_models()

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

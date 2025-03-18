from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif
from module.utils import *
import numpy as np

class DE_DynamicF:
    def __init__(
        self,
        X,
        y,
        size=global_params["size"],
        alpha=global_params["alpha"],
        beta=global_params["beta"],
        CR=0.5,  # 移除固定F参数
        max_FES=global_params["max_FES"],
    ):
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.CR = CR
        self.max_FES = max_FES

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.dimension = X.shape[1]
        self.population = np.zeros((self.size, self.dimension), dtype=int)
        self.x = np.zeros((self.size, self.dimension))
        self.fitness_x = np.zeros(self.size)
        self.FES = 0
        self.global_best_fitness = float("inf")
        self.global_best = np.zeros(self.dimension, dtype=int)
        self.f_best = []
        self.knn = KNeighborsClassifier(n_neighbors=5)
        
        # 新增特征权重计算
        self.feature_weights = mutual_info_classif(self.X_train, self.y_train)
        # 归一化处理
        self.feature_weights = (self.feature_weights - self.feature_weights.min()) / \
                              (self.feature_weights.max() - self.feature_weights.min() + 1e-8)
        
        # 初始化动态F值矩阵
        self.F_matrix = np.zeros((self.size, self.dimension))
        self.update_F_matrix()

    def update_F_matrix(self):
        """根据特征权重动态更新F值矩阵"""
        for d in range(self.dimension):
            if self.feature_weights[d] > 0.8:
                self.F_matrix[:, d] = 0.5
            elif 0.3 <= self.feature_weights[d] <= 0.7:
                self.F_matrix[:, d] = 1.2
            else:
                self.F_matrix[:, d] = 0.0

    def init_solution(self):
        self.population = np.zeros((self.size, self.dimension), dtype=int)
        self.x = np.zeros((self.size, self.dimension))
        self.fitness_x = np.zeros(self.size)
        self.FES = 0
        self.global_best_fitness = float("inf")
        self.global_best = np.zeros(self.dimension, dtype=int)
        self.f_best = []
        self.t = tqdm(total=self.max_FES, desc="DE_DynamicF", bar_format=bar_format)
        
        for i in range(self.size):
            self.x[i] = np.random.rand(self.dimension)
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

    def similarity_selection(self, current_index):
        """相似性导向的基向量选择"""
        current_vector = self.population[current_index]
        similarities = []
        
        # 计算相似性得分
        for i in range(self.size):
            if i == current_index:
                continue
            intersection = np.sum(current_vector & self.population[i])
            union = np.sum(current_vector | self.population[i])
            similarities.append(intersection / (union + 1e-8))
        
        # 轮盘赌选择
        probabilities = np.array(similarities) / (np.sum(similarities) + 1e-8)
        return np.random.choice([i for i in range(self.size) if i != current_index], 
                              p=probabilities)

    def dynamic_mutation(self, i):
        """动态F值变异策略"""
        # 相似性选择基向量
        base_index = self.similarity_selection(i)
        x_base = self.x[base_index]
        
        # 随机选择两个不同个体
        r1, r2 = np.random.choice([j for j in range(self.size) if j not in [i, base_index]], 2, replace=False)
        
        # 应用动态F值
        delta = self.F_matrix[i] * (self.x[r1] - self.x[r2])
        V = x_base + delta
        return np.clip(V, 0, 1)

    def update(self):
        while self.FES < self.max_FES:
            self.t.set_postfix({"solution": self.global_best[:16], 
                              "fitness": f"{self.global_best_fitness:.4f}"})
            
            # 每10代更新一次F值矩阵
            if self.FES % (10*self.size) == 0:
                self.update_F_matrix()
                
            for i in tqdm(range(self.size), desc="种群进化中", leave=False):
                V = self.dynamic_mutation(i)
                U = self.x[i].copy()
                
                # 维度级交叉操作
                for d in range(self.dimension):
                    # 对于F=0的维度直接保留原值
                    if self.F_matrix[i, d] == 0:
                        U[d] = self.x[i, d]
                    elif np.random.rand() < self.CR:
                        U[d] = V[d]
                        
                population_U = (U > 0.5).astype(int)
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
                # 记录中间出现的fitness值
                self.f_best.append(self.global_best_fitness)
                if self.FES >= self.max_FES:
                    return

    def fit(self):
        self.init_solution()
        self.update()
        
        X_train = self.X_train.iloc[:, self.global_best == 1]
        X_test = self.X_test.iloc[:, self.global_best == 1]
        
        if X_train.shape[1] == 0:
            return 0
            
        self.knn.fit(X_train, self.y_train)
        y_pred = self.knn.predict(X_test)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        
        self.t.set_postfix({"accuracy": f"{self.accuracy*100:.2f}%",
                          "solution": self.global_best[:16],
                          "fitness": f"{self.global_best_fitness:.4f}"})
        self.t.close()
        return self.accuracy
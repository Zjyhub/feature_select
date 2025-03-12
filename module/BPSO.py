from sklearn.neighbors import KNeighborsClassifier
from module.utils import *


class BPSO:
    def __init__(
        self,
        X,
        y,
        size=20,
        v_high=6,
        alpha=0.99,
        beta=0.01,
        c1=2.0,
        c2=2.0,
        r=1,
        w=1,
        w_max=0.9,
        w_min=0.4,
        max_FES=1000,
    ):
        """
        初始化BPSO算法对象

        参数:
        X: 特征矩阵，形状为 (样本数量, 特征数量)
        y: 目标类别标签，形状为 (样本数量,)
        size: 粒子群大小，默认值为20
        v_high: 粒子速度最大值，速度范围为 [-v_high, v_high]，默认值为6
        alpha: 惩罚系数，用于控制分类准确率的权重，默认值为0.99
        beta: 惩罚系数，用于控制特征数量的权重，默认值为0.01
        c1: 加速度系数，用于控制个体最优的影响，默认值为2.0
        c2: 加速度系数，用于控制全局最优的影响，默认值为2.0
        r: 选择函数(r为1则使用sigmoid函数，否则使用tanh函数)，默认值为1
        w: 惯性权重，用于控制上一次速度的影响，默认值为1
        w_max: 惯性权重最大值，默认值为0.9
        w_min: 惯性权重最小值，默认值为0.4
        max_FES: 最大评估次数，默认值为1000
        if_obl: 是否使用反转解，默认值为False
        """
        self.size = size
        self.v_high = v_high
        self.alpha = alpha
        self.beta = beta
        self.c1 = c1
        self.c2 = c2
        self.r = r
        self.w = w
        self.w_max = w_max
        self.w_min = w_min
        self.max_FES = max_FES

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.dimension = X.shape[1]  # 特征数量
        self.knn = KNeighborsClassifier(n_neighbors=5)  # 使用k为5的KNN分类器

    # 初始化粒子群
    def init_solution(self):
        self.x = np.zeros((self.size, self.dimension),dtype=int) # 使用一个二维数组来存储粒子群的位置
        self.p_best = np.zeros(
            (self.size, self.dimension),
            dtype=int
        )  # 使用一个二维数组来存储粒子群中每个粒子的历史最佳位置
        self.global_best = np.zeros(self.dimension,dtype=int)  # 使用一个一维数组来存储粒子群的全局最佳位置
        self.v = np.zeros((self.size, self.dimension))  # 使用一个二维数组来存储粒子群的速度
        self.global_best_fitness = float("inf")  # 粒子群的全局最佳适应度初始化为正无穷
        self.p_best_fitness = np.zeros(self.size)  # 使用一个一维数组来存储粒子群中每个粒子的历史最佳适应度
        self.f_best = []  # 存储每次迭代的全局最优适应度值
        self.FES=0
        self.t=tqdm(total=self.max_FES,desc="BPSO",bar_format=bar_format)
        for i in range(self.size):
            # 初始化粒子群的位置和速度,位置初始化为一个随机的二进制向量,速度初始化为一个随机的向量
            self.x[i] = np.random.choice([0, 1], self.dimension)  # 随机生成一个二进制向量
            self.p_best[i] = self.x[i]  # 个体最优位置初始化为当前位置
            f_new = fitness(
                self.alpha,
                self.beta,
                self.dimension,
                self.X_train,
                self.y_train,
                self.x[i],
                self.knn,
            )  # 计算适应度函数值
            self.p_best_fitness[i] = f_new  # 个体最优适应度值初始化为当前适应度函数值

            # 更新全局最优位置
            if f_new < self.global_best_fitness:
                self.global_best = self.p_best[i]
                self.global_best_fitness = f_new

    # 粒子群更新
    def update(self):
        while self.FES<self.max_FES:
            self.t.set_postfix({"solution":self.global_best[:16],"fitness":f"{self.global_best_fitness:.4f}"})
            # 更新惯性权重
            self.w = self.w_max - (self.w_max - self.w_min) * self.FES / self.max_FES
            # 遍历每个粒子，更新每个粒子的位置和速度
            for i in tqdm(range(self.size),desc="种群进化中",leave=False):
                # 更新当前粒子的速度
                self.v[i] = (
                    self.w * self.v[i]
                    + self.c1 * np.random.rand() * (self.p_best[i] - self.x[i])
                    + self.c2 * np.random.rand() * (self.global_best - self.x[i])
                )
                self.v[i] = np.clip(self.v[i], -self.v_high, self.v_high)  # 限制速度范围

                # 更新位置,遍历每个维度
                for j in range(self.dimension):
                    # 如果r为1则使用sigmoid函数，否则使用tanh函数
                    if self.r == 1:
                        self.x[i][j] = sigmoid(self.v[i][j])
                    else:
                        self.x[i][j] = tanh(self.v[i][j])

                    # 如果随机数大于x[i][j]，则x[i][j]取1，否则取0
                    if self.x[i][j] > np.random.rand():
                        self.x[i][j] = 1
                    else:
                        self.x[i][j] = 0

                self.x[i] = np.clip(self.x[i], 0, 1)  # 限制位置范围
                f_new = fitness(
                    self.alpha,
                    self.beta,
                    self.dimension,
                    self.X_train,
                    self.y_train,
                    self.x[i],
                    self.knn,
                )  # 计算当前粒子的适应度函数值

                # 更新个体最优位置
                if f_new < self.p_best_fitness[i]:
                    self.p_best[i] = self.x[i]
                    self.p_best_fitness[i] = f_new

                # 更新全局最优位置
                if f_new < self.global_best_fitness:
                    self.global_best = self.p_best[i]
                    self.global_best_fitness = f_new

                self.FES += 1
                self.t.update(1)
                # 如果评估次数超过最大评估次数，则停止迭代
                if self.FES >= self.max_FES:
                    self.f_best.append(self.global_best_fitness)
                    return
            # 记录每次迭代的全局最优适应度值
            self.f_best.append(self.global_best_fitness)

    # 训练模型,返回全局最优位置
    def fit(self):
        self.init_solution()  # 初始化粒子群
        self.update()  # 粒子群更新

        # 使用knn算法在测试集上进行测试
        X_train = self.X_train.iloc[:,self.global_best==1]
        X_test = self.X_test.iloc[:,self.global_best==1]
        # 如果选择的特征数量为0，则返回0，否则返回在测试集上的准确率
        if X_train.shape[1] == 0:
            return 0
        self.knn.fit(X_train, self.y_train)
        y_pred = self.knn.predict(X_test)
        self.accuracy = accuracy_score(self.y_test, y_pred)
        self.t.set_postfix({"accuracy":f"{self.accuracy*100:.2f}%","solution":self.global_best[:16],"fitness":f"{self.global_best_fitness:.4f}"})
        self.t.close()
        return self.accuracy

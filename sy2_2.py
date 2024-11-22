import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 加载数据集
data = load_breast_cancer()
X = data.data[:, :2]  # 只选择前两个特征
y = data.target  # 标签

# 将数据划分为训练集和测试集（80%训练，20%测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化Logistic回归模型
model = LogisticRegression(max_iter=10000)

# 训练模型
model.fit(X_train, y_train)

# 获取模型系数
theta_0 = model.intercept_[0]
theta_1 = model.coef_[0][0]
theta_2 = model.coef_[0][1]

# 输出模型系数
print(f"θ0 = {theta_0:.4f}")
print(f"θ1 = {theta_1:.4f}")
print(f"θ2 = {theta_2:.4f}")

# 分类直线方程
print(f"分类直线方程： P(y=1|x) = 1 / (1 + exp(-(θ0 + θ1 * x1 + θ2 * x2)))")

# 在二维平面上绘制训练数据和测试数据
plt.figure(figsize=(10, 6))

# 绘制训练数据点，按类别进行颜色区分
plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], color='blue', label='Malignant (0)', alpha=0.5)
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', label='Benign (1)', alpha=0.5)

# 绘制决策边界
xx, yy = np.meshgrid(np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100),
                     np.linspace(X_train[:, 1].min(), X_train[:, 1].max(), 100))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)

# 添加标签和标题
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.legend()

# 显示图形
plt.show()


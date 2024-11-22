import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

# 数据1 (5组数据)
x1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
y1 = np.array([-0.070676, -0.26191242, 0.17673216, 0.46497056, 0.87099123])

# 数据2 (21组数据)
x2 = np.array([0, 0.1*np.pi, 0.2*np.pi, 0.3*np.pi, 0.4*np.pi, 0.5*np.pi, 0.6*np.pi, 0.7*np.pi, 0.8*np.pi,
               0.9*np.pi, 1.0*np.pi, 1.1*np.pi, 1.2*np.pi, 1.3*np.pi, 1.4*np.pi, 1.5*np.pi, 1.6*np.pi, 1.7*np.pi,
               1.8*np.pi, 1.9*np.pi, 2.0*np.pi])

# 这里确保 y2 有 21 个数据点，最后一个数据点与数据1中的第5组相同
y2 = np.array([-0.070676, 0.62069094, 0.9563237, -0.03169173, 2.83525242, -0.26191242, -0.76075376, -1.60192248,
               -0.88072372, -1.8541696, 0.17673216, -1.93444425, 0.20282296, -1.41915353, -1.65908809, 0.46497056,
               -0.24124491, -0.47454724, 0.55812574, 1.9597344, 0.87099123])  # 确保 y2 包含21个值，最后一个值与数据1的第5个一致

# 多项式回归类
class PolynomialModel:
    def __init__(self, degree=2):
        self.degree = degree
        self.model = LinearRegression()

    def fit(self, x, y):
        x_poly = np.vstack([x**i for i in range(self.degree + 1)]).T  # 构造多项式特征
        self.model.fit(x_poly, y)

    def score(self, x, y):
        x_poly = np.vstack([x**i for i in range(self.degree + 1)]).T
        return self.model.score(x_poly, y)

    def get_params(self):
        return self.model.coef_, self.model.intercept_

    def predict(self, x):
        x_poly = np.vstack([x**i for i in range(self.degree + 1)]).T
        return self.model.predict(x_poly)

# 创建一个函数来训练并获取Lasso回归和岭回归的系数
def train_and_get_coefficients(model, x, y, degree):
    x_poly = np.vstack([x**i for i in range(degree + 1)]).T  # 多项式特征
    model.fit(x_poly, y)
    return model.coef_, model.intercept_

# 设置多项式回归的度数
degrees = [2, 3, 4, 5]  # 不同的度数

# 创建Lasso回归模型
lasso = Lasso(alpha=0.1)  # alpha为正则化参数
ridge = Ridge(alpha=0.1)  # alpha为正则化参数

# 遍历不同的degree，训练并打印结果
for degree in degrees:
    print(f"\n==== Degree {degree} ====\n")

    # 多项式回归
    model_poly = PolynomialModel(degree=degree)
    model_poly.fit(x1, y1)
    a1_poly, b1_poly = model_poly.get_params()
    print(f"数据1的多项式回归系数 a0 到 a{degree}: {a1_poly}")
    print(f"数据1的截距 b1: {b1_poly}")
    print(f"数据1的a0到a{degree}（打印系数）:")
    for i in range(degree):
        print(f"a{i} = {a1_poly[i]:.4f}")

    # 数据2的多项式回归
    model_poly.fit(x2, y2)
    a2_poly, b2_poly = model_poly.get_params()
    print(f"数据2的多项式回归系数 a0 到 a{degree}: {a2_poly}")
    print(f"数据2的截距 b2: {b2_poly}")
    print(f"数据2的a0到a{degree}（打印系数）:")
    for i in range(degree):
        print(f"a{i} = {a2_poly[i]:.4f}")

    # Lasso回归
    a1_lasso, b1_lasso = train_and_get_coefficients(lasso, x1, y1, degree)
    print(f"数据1的Lasso回归系数 a0 到 a{degree} (Lasso): {a1_lasso}")
    print(f"数据1的截距 b1 (Lasso): {b1_lasso}")
    print(f"数据1的a0到a{degree}（Lasso）:")
    for i in range(degree):
        print(f"a{i} (Lasso) = {a1_lasso[i]:.4f}")

    # 数据2的Lasso回归
    a2_lasso, b2_lasso = train_and_get_coefficients(lasso, x2, y2, degree)
    print(f"数据2的Lasso回归系数 a0 到 a{degree} (Lasso): {a2_lasso}")
    print(f"数据2的截距 b2 (Lasso): {b2_lasso}")
    print(f"数据2的a0到a{degree}（Lasso）:")
    for i in range(degree):
        print(f"a{i} (Lasso) = {a2_lasso[i]:.4f}")

    # 岭回归
    a1_ridge, b1_ridge = train_and_get_coefficients(ridge, x1, y1, degree)
    print(f"数据1的岭回归系数 a0 到 a{degree} (Ridge): {a1_ridge}")
    print(f"数据1的截距 b1 (Ridge): {b1_ridge}")
    print(f"数据1的a0到a{degree}（Ridge）:")
    for i in range(degree):
        print(f"a{i} (Ridge) = {a1_ridge[i]:.4f}")

    # 数据2的岭回归
    a2_ridge, b2_ridge = train_and_get_coefficients(ridge, x2, y2, degree)
    print(f"数据2的岭回归系数 a0 到 a{degree} (Ridge): {a2_ridge}")
    print(f"数据2的截距 b2 (Ridge): {b2_ridge}")
    print(f"数据2的a0到a{degree}（Ridge）:")
    for i in range(degree):
        print(f"a{i} (Ridge) = {a2_ridge[i]:.4f}")

    # 绘制拟合图
    plt.figure(figsize=(12, 8))

    # 数据1的多项式回归拟合图
    plt.subplot(2, 3, 1)
    plt.scatter(x1, y1, color='blue', label='Data 1')
    plt.plot(x1, model_poly.predict(x1), color='red', label='Polynomial Fit')
    plt.title(f"Degree {degree} - data1 Polynomial")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    # 数据2的多项式回归拟合图
    plt.subplot(2, 3, 2)
    plt.scatter(x2, y2, color='green', label='Data 2')
    plt.plot(x2, model_poly.predict(x2), color='orange', label='Polynomial Fit')
    plt.title(f"Degree {degree} - data2 Polynomial")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    # 数据1的Lasso回归拟合图
    plt.subplot(2, 3, 3)
    plt.scatter(x1, y1, color='blue', label='Data 1')
    plt.plot(x1, lasso.predict(np.vstack([x1**i for i in range(degree+1)]).T), color='red', label='Lasso Fit')
    plt.title(f"Degree {degree} - data1 Lasso")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    # 数据2的Lasso回归拟合图
    plt.subplot(2, 3, 4)
    plt.scatter(x2, y2, color='green', label='Data 2')
    plt.plot(x2, lasso.predict(np.vstack([x2**i for i in range(degree+1)]).T), color='orange', label='Lasso Fit')
    plt.title(f"Degree {degree} - data2 Lasso")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    # 数据1的岭回归拟合图
    plt.subplot(2, 3, 5)
    plt.scatter(x1, y1, color='blue', label='Data 1')
    plt.plot(x1, ridge.predict(np.vstack([x1**i for i in range(degree+1)]).T), color='red', label='Ridge Fit')
    plt.title(f"Degree {degree} - data1 Ridge")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    # 数据2的岭回归拟合图
    plt.subplot(2, 3, 6)
    plt.scatter(x2, y2, color='green', label='Data 2')
    plt.plot(x2, ridge.predict(np.vstack([x2**i for i in range(degree+1)]).T), color='orange', label='Ridge Fit')
    plt.title(f"Degree {degree} - data2 Ridge")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()

    plt.tight_layout()
    plt.show()





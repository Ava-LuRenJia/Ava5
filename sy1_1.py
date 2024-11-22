import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 数据1：原始x和y数据
x_data1 = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4]).reshape(-1, 1)
y_data1 = np.array([1.01, 1.19, 1.42, 1.57, 1.83, 2.58, 3.38, 4.22, 5.01, 5.79])

# 数据2：创建带高斯噪声的数据
x_data2 = np.linspace(0, 10, 20).reshape(-1, 1)
np.random.seed(0)  # 固定随机种子便于复现
noise = np.random.normal(0, 1, x_data2.shape)
y_data2 = 2.5 * x_data2 + 1.5 + noise  # 假设线性关系 y = 2.5x + 1.5，并加入噪声

# 创建并训练模型1（数据1）
lr1 = LinearRegression()
lr1.fit(x_data1, y_data1)
a1 = lr1.coef_[0]
b1 = lr1.intercept_
# 输出数据1的斜率和截距
print(f"数据1的斜率 a1: {a1:.4f}, 截距 b1: {b1:.4f}")

# 创建并训练模型2（数据2）
lr2 = LinearRegression()
lr2.fit(x_data2, y_data2)
a2 = lr2.coef_[0]
b2 = lr2.intercept_
# 输出数据2的斜率和截距
print(f"数据2的斜率 a2: {a2[0]:.4f}, 截距 b2: {b2[0]:.4f}")


# 预测数据1中 x = 3 和 x = 6.5 的 y 值
y_pred_1_x3 = lr1.predict([[3]])[0]
y_pred_1_x6_5 = lr1.predict([[6.5]])[0]
print(f"数据1中 x = 3 的预测 y 值: {y_pred_1_x3:.4f}")
print(f"数据1中 x = 6.5 的预测 y 值: {y_pred_1_x6_5:.4f}")

# 预测数据2中 x = 3 和 x = 9.5 的 y 值
y_pred_2_x3 = lr2.predict([[3]])[0]
y_pred_2_x9_5 = lr2.predict([[9.5]])[0]
print(f"数据2中 x = 3 的预测 y 值: {y_pred_2_x3[0]:.4f}")
print(f"数据2中 x = 9.5 的预测 y 值: {y_pred_2_x9_5[0]:.4f}")

# 计算预测值以绘制拟合线
y_predict_1 = lr1.predict(x_data1)
y_predict_2 = lr2.predict(x_data2)

# 绘制数据1的拟合图
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(x_data1, y_data1, color='blue', label='data1')
plt.plot(x_data1, y_predict_1, color='red', label='Linear regression fitting')
plt.xlabel("x")
plt.ylabel("y")
plt.title("data1")
plt.legend()

# 绘制数据2的拟合图
plt.subplot(1, 2, 2)
plt.scatter(x_data2, y_data2, color='green', label='Noisy data 2')
plt.plot(x_data2, y_predict_2, color='red', label='Linear regression fitting')
plt.xlabel("x")
plt.ylabel("y")
plt.title("data2")
plt.legend()

plt.show()

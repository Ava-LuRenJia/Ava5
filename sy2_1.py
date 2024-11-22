import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 给定数据
x = np.array([0.5, 1.5, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
y = np.array([0, 0.69, 1.10, 1.39, 1.61, 1.79, 1.95, 2.08, 2.20, 2.30])

# 计算t = e^y
t = np.exp(y)

# 使用线性回归拟合y = log(ax)，我们要找到a
# 先计算log(x)
log_x = np.log(x)

# 进行线性回归，拟合log(x)和y的关系
slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, y)

# 输出拟合结果
print(f"拟合得到的斜率 (log(a))：{slope}")
print(f"拟合得到的截距：{intercept}")
a = np.exp(slope)  # 从log(a)得到a
print(f"计算得到的a值：{a}")

# 计算每个x对应的t
t_values = np.exp(np.log(a) + slope * log_x)

# 输出结果
for i in range(len(x)):
    print(f"x{i+1} = {x[i]}, t{i+1} = {t[i]:.4f}")

# 输出a值
print(f"\na = {a:.4f}")

# 绘制预测图/分类图

# 绘制实际数据点
plt.scatter(x, t, color='blue', label='Actual data')

# 绘制拟合曲线
# 计算拟合曲线的y值
x_fit = np.linspace(min(x), max(x), 100)
y_fit = np.exp(np.log(a) + slope * np.log(x_fit))

plt.plot(x_fit, y_fit, color='red', label=f'Fitting line: t = {a:.4f} * x^{slope:.4f}')

# 添加标签和标题
plt.xlabel('x')
plt.ylabel('t')
plt.title('Prediction Chart/Classification chart')

# 添加图例
plt.legend()

# 显示图形
plt.show()


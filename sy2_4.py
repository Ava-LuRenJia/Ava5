from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LinearRegression
import numpy as np

# 加载乳腺癌数据集
breast_cancer = load_breast_cancer()

# （1）提取特征列数据
X = breast_cancer.data[:, :2]  # 提取前两列特征作为 X
Y = breast_cancer.target  # 目标变量

# 创建线性回归模型
regression_model = LinearRegression()

# （2）拟合模型，训练回归模型
regression_results = regression_model.fit(X, Y)

# （3）使用模型进行预测，输入进行对数变换
y_predict = regression_results.predict(np.log(X))  # 预测结果

# （4）# 计算参数： θ，即回归模型的系数
theta = regression_model.coef_  # 模型的权重系数
print("模型参数 θ:", theta)

# 打印截距
intercept = regression_model.intercept_
print("模型截距:", intercept)

# 打印预测结果
print("预测结果:", y_predict)

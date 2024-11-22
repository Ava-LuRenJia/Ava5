from sklearn.linear_model import LinearRegression, Ridge, MultiTaskLassoCV
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 加载数据
def load_data():
    # 读取Excel文件数据
    data = pd.read_excel("data.xlsx")
    # 删除第一列（时间列），保留后面的自变量和目标变量
    X = data.iloc[:, 1:-4].values  # 自变量（从第二列到倒数第四列）
    y = data.iloc[:, -4:].values   # 目标变量（最后四列成分）

    # 将数据分割为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# 训练和评估模型
def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)  # 训练模型
    r2_score = model.score(X_test, y_test)  # 计算 R2 分数
    y_pred = model.predict(X_test)  # 预测值
    return r2_score, y_pred

# 绘制拟合图
def plot_fitted_results(y_test, y_pred, target_name):
    plt.figure(figsize=(10, 6))
    for i in range(y_test.shape[1]):
        plt.subplot(2, 2, i+1)
        plt.plot(y_test[:, i], label="True values")
        plt.plot(y_pred[:, i], label="Predicted values")
        plt.title(f'{target_name} - Component {i+1}')
        plt.xlabel('Sample index')
        plt.ylabel('Value')
        plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # 加载数据
    X_train, X_test, y_train, y_test = load_data()

    # 实例化回归模型
    linear_reg = LinearRegression()
    Ridge_reg = Ridge(alpha=0.5)
    Lasso_reg = MultiTaskLassoCV(cv=10)  # 使用 MultiTaskLassoCV

    print("图中Component1-4分别代表成品成分C、成品成分Si、成品成分Mn、成品成分Mn/Si比；\n")
    # 训练并评估每个模型
    print("Training LinearRegression...")
    r2_linear, y_pred_linear = train_and_evaluate(linear_reg, X_train, y_train, X_test, y_test)
    print(f"LinearRegression R2 score: {r2_linear}")
    print(f"LinearRegression Mean Squared Error: {mean_squared_error(y_test, y_pred_linear)}")

    print("Training Ridge...")
    r2_ridge, y_pred_ridge = train_and_evaluate(Ridge_reg, X_train, y_train, X_test, y_test)
    print(f"Ridge R2 score: {r2_ridge}")
    print(f"Ridge Mean Squared Error: {mean_squared_error(y_test, y_pred_ridge)}")

    print("Training Lasso...")
    r2_lasso, y_pred_lasso = train_and_evaluate(Lasso_reg, X_train, y_train, X_test, y_test)
    print(f"Lasso R2 score: {r2_lasso}")
    print(f"Lasso Mean Squared Error: {mean_squared_error(y_test, y_pred_lasso)}")

    # 画出拟合图
    plot_fitted_results(y_test, y_pred_linear, 'LinearRegression')
    plot_fitted_results(y_test, y_pred_ridge, 'Ridge')
    plot_fitted_results(y_test, y_pred_lasso, 'Lasso')

if __name__ == "__main__":
    main()


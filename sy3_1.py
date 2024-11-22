import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


# 定义感知机的结构图
def plot_perceptron():
    # 创建一个有向图
    G = nx.DiGraph()

    # 定义感知机的层次
    input_nodes = ['x1', 'x2']
    hidden_nodes = ['h1', 'h2']
    output_node = ['o1']

    # 添加节点到图中
    G.add_nodes_from(input_nodes, layer='input')
    G.add_nodes_from(hidden_nodes, layer='hidden')
    G.add_nodes_from(output_node, layer='output')

    # 添加边（连接感知机层之间的节点）
    # 输入层到隐藏层
    for i in input_nodes:
        for h in hidden_nodes:
            G.add_edge(i, h)

    # 隐藏层到输出层
    for h in hidden_nodes:
        for o in output_node:
            G.add_edge(h, o)

    # 设置节点位置
    pos = {}
    pos.update((node, (1, idx)) for idx, node in enumerate(input_nodes))  # 输入层
    pos.update((node, (2, idx)) for idx, node in enumerate(hidden_nodes))  # 隐藏层
    pos.update((node, (3, 0)) for node in output_node)  # 输出层

    # 绘制图形
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=15, font_weight="bold",
            arrows=True)

    # 显示图形
    plt.title('Two-Layer Perceptron Structure for XOR Problem', fontsize=16)
    plt.show()


# 画出感知机结构图
plot_perceptron()

# 解决异或问题的感知机模型
# 定义 XOR 数据集
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])  # XOR 输出

# 初始化参数
input_dim = 2  # 输入层大小
hidden_dim = 2  # 隐藏层大小
output_dim = 1  # 输出层大小

# 初始化权重和偏置
W1 = np.random.randn(input_dim, hidden_dim)  # 输入层到隐藏层的权重
b1 = np.zeros((1, hidden_dim))  # 隐藏层的偏置
W2 = np.random.randn(hidden_dim, output_dim)  # 隐藏层到输出层的权重
b2 = np.zeros((1, output_dim))  # 输出层的偏置


# 激活函数：Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 激活函数的导数
def sigmoid_derivative(x):
    return x * (1 - x)


# 训练参数
epochs = 10000  # 训练轮数
learning_rate = 0.1  # 学习率

# 训练过程
for epoch in range(epochs):
    # 前向传播
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)
    output_input = np.dot(hidden_output, W2) + b2
    output_output = sigmoid(output_input)

    # 计算误差
    error = y - output_output

    # 反向传播
    output_delta = error * sigmoid_derivative(output_output)
    hidden_delta = output_delta.dot(W2.T) * sigmoid_derivative(hidden_output)

    # 更新权重和偏置
    W2 += hidden_output.T.dot(output_delta) * learning_rate
    b2 += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
    W1 += X.T.dot(hidden_delta) * learning_rate
    b1 += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    # 每1000轮输出一次训练误差
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Error: {np.mean(np.abs(error))}')

# 测试模型
print("\n预测结果：")
predictions = sigmoid(np.dot(sigmoid(np.dot(X, W1) + b1), W2) + b2)
for i in range(len(X)):
    print(f"输入: {X[i]} -> 预测输出: {predictions[i]} (真实值: {y[i]})")






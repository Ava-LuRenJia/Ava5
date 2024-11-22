import tensorflow as tf
from keras import layers, models
# 使用 TensorFlow 直接加载 MNIST 数据集
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 数据归一化
train_images = train_images / 255.0
test_images = test_images / 255.0

# 定义输入和输出
input_dim = 784  # 28x28图片展平后是784个像素
output_dim = 10  # 10个数字类别

# 神经网络的超参数
n_hidden_1 = 256  # 第一层隐藏单元数
learning_rate = 0.001  # 学习率
epochs = 10  # 训练周期数
batch_size = 64  # 每批次数据量

# 展平图片数据
train_images = train_images.reshape(-1, input_dim)
test_images = test_images.reshape(-1, input_dim)

# 创建模型
model = models.Sequential([
    layers.InputLayer(input_shape=(input_dim,)),  # 输入层，784个神经元
    layers.Dense(n_hidden_1, activation='sigmoid'),  # 第一隐藏层，256个神经元
    layers.Dense(output_dim, activation='softmax')  # 输出层，10个神经元，对应10个类别
])

# 损失函数定义（交叉熵损失）
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# 优化算法定义（Adam优化器）
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# 编译模型
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# 训练模型
model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(test_images, test_labels))

# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test accuracy: {test_acc}")
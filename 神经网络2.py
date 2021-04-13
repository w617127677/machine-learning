import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
tf.compat.v1.disable_eager_execution()
# 导入或者随机定义训练的数据 x 和 y：
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3
# 先定义出参数 Weights，biases，拟合公式 y，误差公式 loss：
Weights = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
y = Weights*x_data + biases
loss = tf.reduce_mean(tf.square(y-y_data))
# 选择 Gradient Descent 这个最基本的 Optimizer：
optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.5)
# 神经网络的 key idea，就是让 loss 达到最小：
train = optimizer.minimize(loss)
# 前面是定义，在运行模型前先要初始化所有变量：
init = tf.compat.v1.global_variables_initializer()
# 接下来把结构激活，sesseion像一个指针指向要处理的地方：
sess = tf.compat.v1.Session()
# init 就被激活了，不要忘记激活：
sess.run(init)
for step in range(201):

    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))





# import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# from __future__ import print_function
tf.disable_v2_behavior()
# 添加层
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
# 1.训练的数据

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]

noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# print(y_data)
# 2.定义节点准备接收数据
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
# add hidden layer
# 3.定义神经层：隐藏层和预测层
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer
prediction = add_layer(l1, 10, 1, activation_function=None)
# 4.定义 loss 表达式

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
# 5.选择 optimizer 使 loss 达到最小
# 这一行定义了用什么方式去减少 loss，学习率是 0.1
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


sess = tf.Session()

# important step 对所有变量进行初始化
init = tf.initialize_all_variables()
# init = tf.global_variables_initializer()
sess.run(init)

# plot the real data
plt.scatter(x_data,y_data)


# plt.show()
saver = tf.train.Saver(max_to_keep = 4 , keep_checkpoint_every_n_hours = 1 )

for i in range(100):
    # training
    gg = sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    # if i % 50 == 0:
        # to visualize the result and improvement
        # try:
        #     # print(lines[0])
        #     ax.lines.remove(lines[0])
        # except Exception:
        #     pass
    prediction_value = sess.run(prediction, feed_dict={xs: x_data})
    # writer.add_summary(prediction)
    # plot the prediction
    plt.cla()
    plt.scatter(x_data, y_data)
    lines = plt.plot(x_data, prediction_value,'red', lw=5)

    # plt.pause(1)
    plt.ion()
    plt.show()
    print(prediction)

    print(1-sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
saver.save(sess, 'my_test_model' )
# plt.ion()
from __future__ import print_function  # 强制使用python3版本，不管python使用的是什么版本
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data  # 导入mnist库
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # 如果没有mnist数据就进行下载，使用one_hot编码


def add_layer(inputs, in_size, out_size, activation_function=None, ):  # 神经网络函数
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    return outputs


def compute_accuracy(v_xs, v_ys):  # 计算精度函数
    global prediction  # 在函数里定义全局变量
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys,
                                                                 1))  # 函数tf.equal(x,y,name=None)对比x与y矩阵/向量中相等的元素，相等的返回True，不相等返回False，返回的矩阵/向量的维度与x相同；tf.argmax()返回最大值对应的下标（1表示每一列中的，0表示每一行）
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                      tf.float32))  # tf.cast()类型转换函数，将correct_prediction转换成float32类型，并对correct_prediction求平均值得到arruracy
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])  # 28x28    #输入为N个图片，每个图片由28x28 个像素点组成
ys = tf.placeholder(tf.float32, [None, 10])  # 输出N个数据 ，每张图片识别一个数字 0-9共十种

# add output layer
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)  # 输入784，输出10，激励函数使用softmax

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[
                                                  1]))  # loss  loss函数（即最优化目标函数）选用交叉熵函数。交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，它们的交叉熵等于零。
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)  # 梯度下降法，学习速率是0.5

sess = tf.Session()
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)  # 初始化模型的参数

for i in range(10000):  # 训练
    batch_xs, batch_ys = mnist.train.next_batch(100)  # 每次采用100个图片进行训练，避免数据过大，训练太慢
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))  # 测试集，images是输入，labels是标签
import numpy as np
from sklearn.datasets import fetch_california_housing
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
housing = fetch_california_housing()
m, n = housing.data.shape#m为行 n为列
# # np.c_按colunm来组合array
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
# np.ones((m, 1))m行为1
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")

y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
with tf.Session() as sess:
    theta_value = theta.eval()
print(theta_value)
# tf.matrix_diag 对角阵
# tf.matrix_inverse 逆矩阵


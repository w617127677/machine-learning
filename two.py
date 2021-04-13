import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
from tensorflow import keras
import numpy as np

fashion_mnist = keras.datasets.fashion_mnist
mp = "G://机器学习1/iris_model1.h5"
model = load_model(mp)

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
predictions = model.predict(test_images)

error = []
for i in range(len(predictions)):
    if np.argmax(predictions[i]) != test_labels[i]:
        error.append(i)
# print(len(predictions),len(error))
print(error)
#
#
# print(np.argmax(predictions[19]),test_labels[19])
for i in range(100):
    plt.subplot(10, 10, i + 1)
    plt.grid(False)
    plt.imshow(test_images[error[i]], cmap=plt.cm.binary)
    plt.xlabel((np.argmax(predictions[error[i]]),test_labels[error[i]]))
plt.show()
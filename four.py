import os
import warnings
warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import numpy as np
from keras.models import load_model
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)
# train_examples_batch, train_labels_batch = next(iter(test_data.batch(10)))
# print(train_examples_batch,train_labels_batch)



embedding = "https://hub.tensorflow.google.cn/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)
# hub_layer(train_examples_batch[:3])
model = tf.keras.Sequential([
hub_layer,
tf.keras.layers.Dense(16, activation='relu'),
tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)
results = model.evaluate(test_data.batch(512), verbose=2)
for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))
pre_dict = model.predict(test_data.batch(512))
print(np.rint(pre_dict)[0])
print(next(iter(test_data.batch(512)))[1][0])
error = []
# for i in range(100):
#     if np.rint(pre_dict)[0][i][0] !=next(iter(test_data.batch(512)))[1][i]:
#         error.append(i)
# print(error)




# [[-0.34040046]
#  [ 0.9586423 ]
#  [-1.1494484 ]
#  ...
#  [-4.9193287 ]
#  [ 3.4362411 ]
#  [ 3.9776344 ]]

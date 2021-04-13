import os
import warnings
warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow import keras
import numpy as np
from keras.models import load_model
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print(train_data[0])
train_data = keras.preprocessing.sequence.pad_sequences(train_data,maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,maxlen=256)
vocab_size = 10000
model = load_model('G://机器学习1/iris_model2.h5')
# model = keras.Sequential([keras.layers.Embedding(vocab_size, 16),
#                          keras.layers.GlobalAveragePooling1D(),
#                          keras.layers.Dense(16, activation='relu'),
#                          keras.layers.Dense(1, activation='sigmoid')])
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# x_val = train_data[:10000]
# partial_x_train = train_data[10000:]
# y_val = train_labels[:10000]
# partial_y_train = train_labels[10000:]
# history = model.fit(partial_x_train,
#                     partial_y_train,
#                     epochs=40,
#                     batch_size=512,
#                     validation_data=(x_val, y_val),
#                     verbose=1)
results = model.evaluate(test_data,  test_labels, verbose=2)
# model.save("G://机器学习1/iris_model2.h5")
print(results)
import numpy as np
import theano
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Dropout
from keras.layers import LSTM, SpatialDropout1D
from keras.layers import Input, LSTM, Dense
from keras.models import Model
import scipy.io
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from scipy.signal import welch
from keras import utils
from keras.callbacks import EarlyStopping
from keras.utils import plot_model

#################################DATA#######################################

import json
with open("sixDEM_data_file.json", "r") as read_file:
    data = json.load(read_file)

###############################################################################################

model = Sequential()
model.add(Dense(20, activation='relu', input_dim=6))
#model.add(keras.layers.BatchNormalization())
#model.add(Dropout(0.1))
model.add(Dense(40, activation='relu'))
#model.add(Dropout(0.1))
model.add(Dense(20, activation='relu'))
model.add(Dense(6, activation='relu'))

###################################################################################################

from sklearn.metrics import mean_squared_error
#mean_squared_error = mean_squared_error(welch(y_all)[0],welch(x_all)[0])
early_stopping_callback = EarlyStopping(monitor='val_acc', patience=3)
#tb_callbacks = [TensorBoard(log_dir='tb_logs1', histogram_freq=1, write_images=True)]

model.compile(optimizer='adam', loss = 'mean_squared_error',metrics=['accuracy'])

history = model.fit(np.asarray(data[0]), np.asarray(data[1]), epochs=10, validation_split=0.2, callbacks=[early_stopping_callback])#, callbacks=[tb_callbacks, early_stopping_callback])
"""
plt.plot(history.history['loss'], 
         label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_loss'], 
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()"""

model.evaluate(x=np.asarray(data[2]), y=np.asarray(data[3]))


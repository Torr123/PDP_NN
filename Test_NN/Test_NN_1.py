import numpy as np
import theano
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Dropout
#from keras.layers import LSTM, SpatialDropout1D
from keras.layers import Input, LSTM, Dense
#from keras.models import Model

#################################DATA#######################################

from json import load
with open("sixDEM_data_file.json", "r") as read_file:
    data = load(read_file)

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

#from sklearn.metrics import mean_squared_error
#mean_squared_error = mean_squared_error(welch(y_all)[0],welch(x_all)[0])
#early_stopping_callback = EarlyStopping(monitor='val_acc', patience=3)
#tb_callbacks = [TensorBoard(log_dir='tb_logs1', histogram_freq=1, write_images=True)]

model.compile(optimizer='adam', loss = 'mean_squared_error',metrics=['accuracy'])

model.fit(np.asarray(data[0]), np.asarray(data[1]), epochs=10, validation_split=0.2, callbacks=[early_stopping_callback])

model.evaluate(x=np.asarray(data[2]), y=np.asarray(data[3]))


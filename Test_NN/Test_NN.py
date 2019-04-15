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

############################DATA################################

data_dct = scipy.io.loadmat("../Data/Data2MSU.mat")
X = data_dct['PDin'][0]
Y = data_dct['PDout'][0]
x_train = X[:(X.shape[0]*2)//3]
y_train = Y[:(Y.shape[0]*2)//3]
x_test = X[(X.shape[0]*2)//3:X.shape[0]]
y_test = Y[(Y.shape[0]*2)//3:Y.shape[0]]
x_train_real = x_train.real
x_train_imag = x_train.imag
y_train_real = y_train.real
y_train_imag = y_train.imag
#VAR
FROM = 100
TO = 60
MID = 80
MULT = 3
FROM *= MULT
TO *= MULT
MID *= MULT
FR_ML = 300
FR_MZ = 800
FR_LZ = 400
#FUNC
def arct(al, re, im, train):
    al = []
    for i in range(0, len(train)):
        al.append(np.hstack((FR_MZ if np.angle(complex(re[i],im[i])) > 0 else FR_LZ, 
                                np.angle(complex(re[i],im[i]))*FR_ML, 
                                abs(re[i]), 
                                FROM if re[i] > 0 else TO, 
                                abs(im[i]), 
                                FROM if im[i] > 0 else TO)))
    return al
    
x_all = []
x_all = arct(x_all,x_train_real,x_train_imag,x_train)
y_all = []
y_all = arct(y_all,y_train_real,y_train_imag,y_train)

##########
y_all = np.asarray(y_all)
x_all = np.asarray(x_all)

#TEST

x_test_real = x_test.real
x_test_imag = x_test.imag

y_test_real = y_test.real
y_test_imag = y_test.imag
x_all_test = []
x_all_test = arct(x_all_test,x_test_real,x_test_imag,x_test)
y_all_test = []
y_all_test = arct(y_all_test,y_test_real,y_test_imag,y_test)
    
y_all_test = np.asarray(y_all_test)
x_all_test = np.asarray(x_all_test)

def pwelch(Xarr, N=2048):
    F, S = welch(Xarr.ravel().real + 1j*Xarr.ravel().imag, fs=2.0, window='hanning', nperseg=N, noverlap=None, nfft=None, detrend='constant', return_onesided=False, scaling='density', axis=-1)
    F = np.hstack([F[:int(F.shape[0]/2):], F[int(F.shape[0]/2)::]+2])
    S = np.hstack([S[:int(S.shape[0]/2):], S[int(S.shape[0]/2)::]])
    return F, S

  
np.random.seed(48)

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

history = model.fit(x_all, y_all, epochs=10, validation_split=0.2, callbacks=[early_stopping_callback])#, callbacks=[tb_callbacks, early_stopping_callback])

plt.plot(history.history['loss'], 
         label='Доля верных ответов на обучающем наборе')
plt.plot(history.history['val_loss'], 
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()

model.evaluate(x=x_all_test, y=y_all_test)

y_p = model.predict(x=x_all_test)

y_p_w = []

for i in range(0,len(y_p)):
  if y_p[i][1+2] < MID:
    y_p[i][0+2]*=-1 
  if y_p[i][3+2] < MID:
    y_p[i][2+2]*=-1
  y_p_w.append(complex(y_p[i][0+2], y_p[i][2+2]))

y_p_w = np.asarray(y_p_w)

plt.plot(welch(y_p_w)[0],welch(y_p_w)[1])
plt.plot(welch(x_test)[0],welch(x_test)[1], color = 'Red')
plt.plot(welch(y_test)[0],welch(y_test)[1], color = 'Orange')

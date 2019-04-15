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

#########################################DATA##################################################

data_dct = scipy.io.loadmat("../Data/Data2MSU.mat")
X = data_dct['PDin'][0]
Y = data_dct['PDout'][0]
x_train = X[:X.shape[0]//50]
y_train = Y[:Y.shape[0]//50]
x_test = X[X.shape[0]//50:X.shape[0]//25]
y_test = Y[Y.shape[0]//50:X.shape[0]//25]
x_train_real = x_train.real
x_train_imag = x_train.imag
y_train_real = y_train.real
y_train_imag = y_train.imag
x_test_real = x_test.real
x_test_imag = x_test.imag
y_test_real = y_test.real
y_test_imag = y_test.imag
#VAR
#FUNC

def pwelch(Xarr, N=2048):
    F, S = welch(Xarr.ravel().real + 1j*Xarr.ravel().imag, fs=2.0, window='hanning', nperseg=N, noverlap=None, nfft=None, detrend='constant', return_onesided=False, scaling='density', axis=-1)
    F = np.hstack([F[:int(F.shape[0]/2):], F[int(F.shape[0]/2)::]+2])
    S = np.hstack([S[:int(S.shape[0]/2):], S[int(S.shape[0]/2)::]])
    return F, S

  
np.random.seed(48)

######################################################BATCH_GEN##############################################

MMM = 2048

def gen_bat(x, M, i):
    if i+M > len(x):
        return (np.vstack((x[i:len(x)],np.zeros((M+i-len(x),2))))).flatten().tolist()
    if i < M:
        x_batch = np.vstack((np.zeros((M-i,2)),x[0:i]))
        return x_batch.flatten().tolist()
    if i >= M:
        x_batch =  x[i:M+i]
        return x_batch.flatten().tolist()
    
############################################################TRAIN_B##########################################

hs_x = []

for i in range(0, len(x_train)):
    hs_x.append(np.hstack((x_train_real[i], x_train_imag[i])))
    
hs_x = np.asarray(hs_x)

x_all = []

for i in range(0, len(x_train)):
    x_all.append(gen_bat(hs_x, MMM, i))

    
y_all = []
for i in range(0, len(y_train)):
    y_all.append(np.hstack((y_train_real[i], y_train_imag[i])))
    

y_all = np.asarray(y_all)
x_all = np.asarray(x_all)

hs_x.shape

#############################################################TEST_B##############################################

hs_x = []

for i in range(0, len(x_test)):
    hs_x.append(np.hstack((x_test_real[i], x_test_imag[i])))
    
hs_x = np.asarray(hs_x)

x_all_test = []

for i in range(0, len(x_test)):
    x_all_test.append(gen_bat(hs_x, MMM, i))

    
y_all_test = []
for i in range(0, len(y_test)):
    y_all_test.append(np.hstack((y_test_real[i], y_test_imag[i])))
    

y_all_test = np.asarray(y_all_test)
x_all_test = np.asarray(x_all_test)

hs_x.shape
################################################################MODEL###############################################

model = Sequential()
model.add(Dense(4096, activation='linear', input_dim=4096)),
model.add(Dense(2048, activation='linear'))
model.add(Dense(1024, activation='linear'))
model.add(Dense(2, activation='linear'))

##################################################################FIT################################################

early_stopping_callback = EarlyStopping(monitor='val_acc', patience=3)
  
model.compile(optimizer='adam', loss='mse',metrics=['mse', 'accuracy'])

history = model.fit(x_all, y_all, epochs=30, validation_split=0.2)


plt.plot(history.history['loss'], 
         label='Лосс на обучающем наборе')
plt.plot(history.history['val_loss'], 
         label='Лосс ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Лосс')
plt.legend()
plt.show()

########################################################################EVAL###########################################

model.evaluate(x=x_all_test, y=y_all_test)

y_p = model.predict(x=x_all_test)

y_p_w = []
for i in range(0, len(y_p)):
    y_p_w.append(complex(y_p[i][0], y_p[i][1]))
    
y_p_w = np.asarray(y_p_w)

plt.plot(pwelch(y_p_w)[0],pwelch(y_p_w)[1])
plt.plot(pwelch(y_test)[0],pwelch(y_test)[1], color = 'Orange')





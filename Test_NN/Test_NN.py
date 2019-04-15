import numpy as np
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


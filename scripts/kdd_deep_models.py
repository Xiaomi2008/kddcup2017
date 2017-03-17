#kdd_deep_models
from keras.layers import merge, Dropout, Dense, Lambda, Flatten, Activation
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers.convolutional import MaxPooling2D, Convolution3D, Convolution2D, AveragePooling2D, ZeroPadding2D, ZeroPadding3D, UpSampling2D, Deconvolution2D, AtrousConvolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape, Permute, Dense
from keras.layers.recurrent import LSTM,GRU
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.extra_conv_recurrent import DeconvLSTM2D, ConvGRU2D, DeconvGRU2D
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.SpatialPyramidPooling import SpatialPyramidPooling

from keras import backend as K
from keras.models import Model
def kdd_model(Input):
	conv1=Convolution2D(64, 75, 1 activation='relu', border_mode='valid')(Input)
	conv2=Convolution2D(128, 1, 3, activation='relu', border_mode='same')(conv1)
	conv3=Convolution2D(128, 1, 3, activation='relu', border_mode='same')(conv2)
	d=Dense(512)(conv3)
	out =Dense(24)(d)
	return out
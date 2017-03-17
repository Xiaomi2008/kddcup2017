#kdd_deep_models
from keras.layers import merge, Dropout, Dense, Lambda, Flatten, Activation
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers.convolutional import MaxPooling2D, Convolution3D, Convolution2D, AveragePooling2D, ZeroPadding2D, ZeroPadding3D, UpSampling2D, Deconvolution2D, AtrousConvolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape, Permute, Dense, Dropout
from keras.layers.recurrent import LSTM,GRU
from keras.layers import Input
# from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras import backend as K
from keras.models import Model
from keras.layers.normalization import BatchNormalization
def kdd_model(input_shape):
	ip = Input(shape=input_shape)
	conv1=Convolution2D(32, (1, 75), activation='relu', padding='valid')(ip)
	conv1=BatchNormalization()(conv1)
	conv1=Dropout(0.5)(conv1)
	conv2=Convolution2D(64, (3, 1), activation='relu', padding='same')(conv1)
	conv2=BatchNormalization()(conv2)
	conv3=Convolution2D(64, (3, 1), activation='relu', padding='same')(conv2)
	conv3=BatchNormalization()(conv3)
	ft=Flatten()(conv3)
	d=Dense(256)(ft)
	d=Dropout(0.5)(d)
	# print(d)
	n =input_shape[0]
	print(n)
	out =Dense(n)(d)
	return ip,out
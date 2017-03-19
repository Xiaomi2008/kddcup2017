#kdd_deep_models
from keras.layers import merge, Dropout, Dense, Lambda, Flatten, Activation
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers.convolutional import MaxPooling1D,Convolution1D,MaxPooling2D, Convolution3D, Convolution2D, AveragePooling2D, ZeroPadding2D, ZeroPadding3D, UpSampling2D, Deconvolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape, Permute, Dense, Dropout
from keras.layers.recurrent import LSTM,GRU
from keras.layers import Input
# from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras import backend as K
from keras.models import Model
from keras.layers.normalization import BatchNormalization
def kdd_model(input_shape,output_shape):
	ip = Input(shape=input_shape)
	print (input_shape)
	conv1=Convolution2D(48, (1, input_shape[1]), activation='relu', padding='valid')(ip)
	conv1=BatchNormalization()(conv1)
	# conv1=Dropout(0.5)(conv1)
	# conv2=Convolution1D(32, 2, activation='relu', padding='valid')(conv1)
	# conv2=BatchNormalization()(conv2)
	# conv3=Convolution1D(32, 2, activation='relu', padding='valid')(conv2)
	# conv3=BatchNormalization()(conv3)
	# conv4=Convolution1D(32, 2, activation='relu', padding='valid')(conv3)
	# conv4=BatchNormalization()(conv4)
	# conv5=Convolution1D(32, 2, activation='relu', padding='valid')(conv4)
	# conv5=BatchNormalization()(conv5)
	conv2=Convolution2D(32, (2, 1), activation='relu', padding='valid')(conv1)
	conv2=BatchNormalization()(conv2)
	conv3=Convolution2D(32, (2, 1), activation='relu', padding='valid')(conv2)
	conv3=BatchNormalization()(conv3)
	pool1=MaxPooling2D(pool_size=(2,1),strides=(2,1))(conv3)
	conv4=Convolution2D(64, (3, 1), activation='relu', padding='valid')(pool1)
	conv4=BatchNormalization()(conv4)
	conv5=Convolution2D(64, (3, 1), activation='relu', padding='valid')(conv4)
	conv5=BatchNormalization()(conv5)
	ft=Flatten()(conv5)
	d=Dense(256)(ft)
	d=BatchNormalization()(d)
	d=Dropout(0.5)(d)
	# print(d)
	n =output_shape[0]
	print(n)
	out =Dense(n)(d)
	return ip,out
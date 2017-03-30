#kdd_deep_models
from keras.layers import merge, Dropout, Dense, Lambda, Flatten, Activation
from keras.layers.pooling import GlobalMaxPooling2D
from keras.layers.convolutional import MaxPooling1D,Convolution1D,MaxPooling2D, Convolution3D, Convolution2D, SeparableConv2D, AveragePooling2D, ZeroPadding2D, ZeroPadding3D, UpSampling2D, Deconvolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape, Permute, Dense, Dropout
from keras.layers.recurrent import LSTM,GRU
from keras.layers.embeddings import Embedding
from keras.layers import Input
from keras import regularizers
# from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras import backend as K
from keras.models import Model
from keras.layers.merge import add, multiply, concatenate
from keras.layers.normalization import BatchNormalization
def kdd_LSTM_1(input_shape,output_shape,number_of_input=1):

	time_step =input_shape[0]
	data_dim  =input_shape[1]
	ips=[]
	
	x_f_lstm_out_list =[]
	for i in range(number_of_input):
		ip=Input(shape=input_shape)
		ips.append(ip)
		x_f_lstm_out_list.append(LSTM(256,return_sequences=True)(ip))
	if number_of_input>2:
		concat=concatenate(x_f_lstm_out_list,axis=-1)
		x2=LSTM(512,return_sequences=True)(concat)
	else:
		x2=LSTM(512,return_sequences=True)(x_f_lstm_out_list[0])
	x3=LSTM(512,return_sequences=False)(x2)
	n=output_shape[0]
	d=Dense(n)(x3)
	# d2=Dense(n)(Flatten()(d))
	return ips ,d
def kdd_LSTM_2(input_shape,output_shape,number_of_input=1):

	time_step =input_shape[0]
	data_dim  =input_shape[1]
	ips=[]
	
	x_f_lstm_out_list =[]
	for i in range(number_of_input):
		ip=Input(shape=input_shape)
		ips.append(ip)
		x_f_lstm_out_list.append(LSTM(256,return_sequences=True)(ip))
	if number_of_input>2:
		concat=concatenate(x_f_lstm_out_list,axis=-1)
		x2=LSTM(512,return_sequences=True)(concat)
	else:
		x2=LSTM(512,return_sequences=True)(x_f_lstm_out_list[0])
	x3=LSTM(512,return_sequences=True)(x2)
	n=output_shape[0]
	d=Dense(1)(x3)
	d2=Dense(n)(Flatten()(d))
	return ips ,d2
def kdd_model_new1(input_shape,output_shape,number_of_input=1):
	ip = Input(shape=input_shape)
	print (input_shape)
	conv1=SeparableConv2D(48, (1, input_shape[1]), activation='relu', padding='valid')(ip)
	conv1=Dropout(0.5)(conv1)
	conv1_bn=BatchNormalization()(conv1)
	conv2=Convolution2D(32, (3, 1), activation='relu', padding='same')(conv1_bn)
	conv2=BatchNormalization()(conv2)
	conv3=Convolution2D(32, (3, 1), activation='relu', padding='same')(conv2)
	conv3=BatchNormalization()(conv3)
	# pool1=MaxPooling2D(pool_size=(2,1),strides=(2,1))(conv3)
	
	conv4=Convolution2D(32, (3, 1), activation='relu', padding='same')(conv3)
	conv4=BatchNormalization()(conv4)
	# conv5=Convolution2D(42, (2, 1), activation='relu', padding='same')(conv4)
	# conv5=BatchNormalization()(conv5)
	# pool2=MaxPooling2D(pool_size=(2,1),strides=(2,1))(conv5)

	conv6=Convolution2D(48, (2, 1), activation='relu', padding='same')(conv4)
	# conv6_=BatchNormalization()(conv6)
	conv1_6=add([conv1,conv6])
	conv6_=BatchNormalization()(conv1_6)
	# conv7=Convolution2D(64, (3, 1), activation='relu', padding='same')(conv6)
	# conv7=BatchNormalization()(conv7)

	
	# ft=Flatten()(conv7)
	ft =Flatten()(conv6)
	d=Dense(256)(ft)
	d=BatchNormalization()(d)
	d=Dropout(0.5)(d)
	# print(d)
	n =output_shape[0]
	print(n)
	out =Dense(n,activation='relu')(d)
	return ip,out
def kdd_model_inception(input_shape,output_shape,number_of_input=1):
	ip = Input(shape=input_shape)
	print (input_shape)
	conv1=SeparableConv2D(48, (1, input_shape[1]), activation='relu', padding='valid',kernel_regularizer=regularizers.l1(0.01))(ip)
	# conv1=Dropout(0.5)(conv1)
	conv1_bn=BatchNormalization()(conv1)
	conv2=Convolution2D(32, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv1_bn)
	conv2=BatchNormalization()(conv2)
	conv3=Convolution2D(32, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv2)
	conv3=BatchNormalization()(conv3)
	conv4=Convolution2D(48, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv3)
	conv1_4=add([conv1_bn,conv4])
	conv5=BatchNormalization()(conv1_4)

	pool1=MaxPooling2D(pool_size=(2,1),strides=(2,1))(conv5)
	conv4_branch=Convolution2D(64, (1, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool1)
	conv4=Convolution2D(64, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool1)
	conv4=BatchNormalization()(conv4)
	conv5=Convolution2D(64, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv4)
	conv2_5=add([conv4_branch,conv5])
	conv_last=BatchNormalization()(conv2_5)
	# ft=Flatten()(conv7)
	ft =Flatten()(conv_last)
	d=Dense(128)(ft)
	d=BatchNormalization()(d)
	d=Dropout(0.5)(d)
	# print(d)
	n =output_shape[0]
	print(n)
	out =Dense(n)(d)
	return ip,out
def kdd_model(input_shape,output_shape):
	ip = Input(shape=input_shape)
	print (input_shape)
	conv1=SeparableConv2D(64, (1, input_shape[1]), activation='relu', padding='valid',kernel_regularizer=regularizers.l2(0.01))(ip)
	conv1=Dropout(0.5)(conv1)
	conv1=BatchNormalization()(conv1)
	conv2=Convolution2D(32, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv1)
	conv2=BatchNormalization()(conv2)
	conv3=Convolution2D(32, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv2)
	conv3=BatchNormalization()(conv3)
	pool1=MaxPooling2D(pool_size=(2,1),strides=(2,1))(conv3)
	
	conv4=Convolution2D(64, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool1)
	conv4=BatchNormalization()(conv4)
	conv5=Convolution2D(64, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv4)
	conv5=BatchNormalization()(conv5)
	pool2=MaxPooling2D(pool_size=(2,1),strides=(2,1))(conv5)

	conv6=Convolution2D(128, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool2)
	conv6=BatchNormalization()(conv6)
	conv7=Convolution2D(128, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv6)
	conv7=BatchNormalization()(conv7)
	conv8=Convolution2D(128, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv7)
	conv8=BatchNormalization()(conv8)

	
	ft=Flatten()(conv8)
	d=Dense(256,kernel_regularizer=regularizers.l2(0.01))(ft)
	d=BatchNormalization()(d)
	d=Dropout(0.5)(d)
	# print(d)
	n =output_shape[0]
	print(n)
	out =Dense(n)(d)
	return ip,out
def kdd_gated_model_2(input_shape,output_shape):
	ip = Input(shape=input_shape)
	print (input_shape)
	#em=Embedding(100,100)(ip)
	conv0=SeparableConv2D(128, (1, input_shape[1]), activation='relu', padding='valid',kernel_regularizer=regularizers.l2(0.01))(ip)
	conv0=BatchNormalization()(conv0)
	embeded_tensor=Permute((1,3,2))(conv0)
	conv1=Convolution2D(128, (1, 128), activation='relu', padding='valid',kernel_regularizer=regularizers.l2(0.01))(embeded_tensor)

	# conv1=Dropout(0.5)(conv1)
	conv1=BatchNormalization()(conv1)
	gate_conv1 =Convolution2D(64, (3, 1), activation='sigmoid', padding='same')(conv1)
	conv2=Convolution2D(64, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv1)
	conv2=BatchNormalization()(conv2)
	conv3=Convolution2D(64, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv2)
	conv3=BatchNormalization()(conv3)
	gate_conv2 = multiply([gate_conv1,conv3])

	pool1=MaxPooling2D(pool_size=(2,1),strides=(2,1))(gate_conv2)
	gate_conv3 =Convolution2D(128, (3, 1), activation='sigmoid', padding='same')(pool1)
	conv4=Convolution2D(128, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool1)
	conv4=BatchNormalization()(conv4)
	conv5=Convolution2D(128, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv4)
	conv5=BatchNormalization()(conv5)
	gate_conv3 = multiply([gate_conv3,conv5])
	pool2=MaxPooling2D(pool_size=(3,1),strides=(2,1))(gate_conv3 )

	gate_conv4=Convolution2D(256, (3, 1), activation='sigmoid', padding='same')(pool2)
	conv6=Convolution2D(256, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool2)
	conv6=BatchNormalization()(conv6)
	conv7=Convolution2D(256, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv6)
	conv7=BatchNormalization()(conv7)
	gate_conv4 = multiply([gate_conv4,conv7])

	
	ft=Flatten()(gate_conv4)
	d=Dense(382,kernel_regularizer=regularizers.l2(0.01))(ft)
	d=BatchNormalization()(d)
	d=Dropout(0.5)(d)
	# print(d)
	n =output_shape[0]
	print(n)
	out =Dense(n)(d)
	return ip,out
def kdd_gated_model(input_shape,output_shape,number_of_input=1):
	ip = Input(shape=input_shape)
	print (input_shape)
	#em=Embedding(100,100)(ip)
	conv0=SeparableConv2D(128, (1, input_shape[1]), activation='relu', padding='valid',kernel_regularizer=regularizers.l1(0.01))(ip)
	conv0=Dropout(0.5)(conv0)
	conv0=BatchNormalization()(conv0)
	embeded_tensor=Permute((1,3,2))(conv0)
	conv1=Convolution2D(64, (1, 128), activation='relu', padding='valid',kernel_regularizer=regularizers.l2(0.01))(embeded_tensor)

	# conv1=Dropout(0.5)(conv1)
	conv1=BatchNormalization()(conv1)
	gate_conv1 =Convolution2D(32, (5, 1), activation='sigmoid', padding='same')(conv1)
	conv2=Convolution2D(32, (5, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv1)
	conv2=BatchNormalization()(conv2)
	conv3=Convolution2D(32, (5, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv2)
	conv3=BatchNormalization()(conv3)
	gate_conv2 = multiply([gate_conv1,conv3])

	pool1=MaxPooling2D(pool_size=(2,1),strides=(2,1))(gate_conv2)
	gate_conv3 =Convolution2D(64, (5, 1), activation='sigmoid', padding='same')(pool1)
	conv4=Convolution2D(64, (5, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool1)
	conv4=BatchNormalization()(conv4)
	conv5=Convolution2D(64, (5, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv4)
	conv5=BatchNormalization()(conv5)
	gate_conv3 = multiply([gate_conv3,conv5])
	pool2=MaxPooling2D(pool_size=(2,1),strides=(2,1))(gate_conv3 )

	gate_conv4=Convolution2D(128, (3, 1), activation='sigmoid', padding='same')(pool2)
	conv6=Convolution2D(128, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool2)
	conv6=BatchNormalization()(conv6)
	conv7=Convolution2D(128, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv6)
	conv7=BatchNormalization()(conv7)
	gate_conv4 = multiply([gate_conv4,conv7])

	
	ft=Flatten()(gate_conv4)
	d=Dense(128,kernel_regularizer=regularizers.l2(0.01))(ft)
	d=BatchNormalization()(d)
	d=Dropout(0.5)(d)
	# print(d)
	n =output_shape[0]
	print(n)
	out =Dense(n)(d)
	return ip,out

def kdd_model_old(input_shape,output_shape):
	ip = Input(shape=input_shape)
	print (input_shape)
	conv1=Convolution2D(256, (1, input_shape[1]), activation='relu', padding='valid')(ip)
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
	conv2=Convolution2D(64, (2, 1), activation='relu', padding='valid')(conv1)
	conv2=BatchNormalization()(conv2)
	conv3=Convolution2D(64, (2, 1), activation='relu', padding='valid')(conv2)
	conv3=BatchNormalization()(conv3)
	pool1=MaxPooling2D(pool_size=(2,1),strides=(2,1))(conv3)
	conv4=Convolution2D(128, (3, 1), activation='relu', padding='valid')(pool1)
	conv4=BatchNormalization()(conv4)
	conv5=Convolution2D(128, (3, 1), activation='relu', padding='valid')(conv4)
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
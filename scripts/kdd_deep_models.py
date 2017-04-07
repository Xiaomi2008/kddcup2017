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
from keras.models import Sequential
# def make_model(dense_layer_sizes, filters, kernel_size, pool_size):
#     '''Creates model comprised of 2 convolutional layers followed by dense layers
#     dense_layer_sizes: List of layer sizes.
#         This list has one number for each layer
#     filters: Number of convolutional filters in each convolutional layer
#     kernel_size: Convolutional kernel size
#     pool_size: Size of pooling area for max pooling
#     '''
#     ip = Input(shape=input_shape)
# 	print (input_shape)
# 	conv1=SeparableConv2D(48, (1, input_shape[1]), activation='relu', padding='valid',kernel_regularizer=regularizers.l1(0.01))(ip)
# 	# conv1=Dropout(0.5)(conv1)
# 	conv1_bn=BatchNormalization()(conv1)
# 	conv2=Convolution2D(32, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv1_bn)
# 	conv2=BatchNormalization()(conv2)
# 	conv3=Convolution2D(32, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv2)
# 	conv3=BatchNormalization()(conv3)
# 	conv4=Convolution2D(48, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv3)
# 	conv1_4=add([conv1_bn,conv4])
# 	conv5=BatchNormalization()(conv1_4)

# 	pool1=MaxPooling2D(pool_size=(2,1),strides=(2,1))(conv5)
# 	pool1=Dropout(0.25)(pool1)
# 	conv4_branch=Convolution2D(64, (1, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool1)
# 	conv4=Convolution2D(64, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool1)
# 	conv4=BatchNormalization()(conv4)
# 	conv5=Convolution2D(64, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv4)
# 	conv2_5=add([conv4_branch,conv5])
# 	conv_last=BatchNormalization()(conv2_5)
# 	conv_last=Dropout(0.25)(conv_last)
# 	# ft=Flatten()(conv7)
# 	ft =Flatten()(conv_last)
# 	d=Dense(128)(ft)
# 	d=BatchNormalization()(d)
# 	d=Dropout(0.5)(d)
# 	# print(d)
# 	n =output_shape[0]
# 	print(n)
# 	out =Dense(n)(d)
# 	return ip,out


#     model = Sequential()
#     model.add(SeparableConv2D(filters, (1, input_shape[1]),
#                      padding='valid',
#                      input_shape=input_shape))
#     model.add(Activation('relu'))
#     model.add(Conv2D(filters, kernel_size))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=pool_size))
#     model.add(Dropout(0.25))

#     model.add(Flatten())
#     for layer_size in dense_layer_sizes:
#         model.add(Dense(layer_size))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(num_classes))
#     model.add(Activation('softmax'))

#     model.compile(loss='categorical_crossentropy',
#                   optimizer='adadelta',
#                   metrics=['accuracy'])

#     return model
def create_model(optimizer='rmsprop', init='glorot_uniform',branch='gate',
				kernel_size=3,number_ch_in_first_block=16,num_blocks=2, 
				layer_in_blocks =3,
				ch_increase_ratio_in_blocks=2, ):
	# create model
	model = Sequential()
	model.add(SeparableConv2D(48,(1,124) , activation='relu', padding='valid',input_shape=(1,124, 60),kernel_regularizer=regularizers.l1(0.01)))
	for i in range(layer_in_blocks):
		model.add(Conv2D(number_ch_in_first_block, (kernel_size,1),kernel_initializer=init, activation='relu'))
	model.compile(optimizer=optimizer, loss=losses.mape, metrics=['MAPE'])
	return model
def sk_learn_grid_search_model():
	pass

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
		# ip=Input(shape=input_shape)
		ip=Input(batch_shape=(1,time_step,data_dim))
		ips.append(ip)
		x_f_lstm_out_list.append(LSTM(64,return_sequences=True,stateful=True)(ip))
	if number_of_input>2:
		concat=concatenate(x_f_lstm_out_list,axis=-1)
		x2=LSTM(64,return_sequences=True,stateful=True)(concat)
	else:
		x2=LSTM(64,return_sequences=True,stateful=True)(x_f_lstm_out_list[0])
	x3=LSTM(64,return_sequences=True,stateful=True)(x2)
	n=output_shape[0]
	d=Dense(1)(x3)
	d2=Dense(n)(Flatten()(d))
	return ips ,d2
def kdd_LSTM_bidirection(input_shape,output_shape,number_of_input=1):

	time_step =input_shape[0]
	data_dim  =input_shape[1]
	ips=[]
	
	x_f_lstm_out_list =[]
	for i in range(number_of_input):
		ip=Input(shape=input_shape)
		ips.append(ip)
		x_f_lstm_out_list.append(Bidirectional(LSTM(32,return_sequences=True))(ip))
	if number_of_input>2:
		concat=concatenate(x_f_lstm_out_list,axis=-1)
		x2=Bidirectional(LSTM(10,return_sequences=True))(concat)
	else:
		x2=Bidirectional(LSTM(10,return_sequences=True))(x_f_lstm_out_list[0])
	x3=Bidirectional(LSTM(10,return_sequences=True))(x2)
	x4=Bidirectional(LSTM(10,return_sequences=True))(x3)
	x5=Bidirectional(LSTM(10,return_sequences=True))(x4)
	n=output_shape[0]
	d=Dense(1)(x5)
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
def kdd_model_simple_gated_condition_prob(input_shape,output_shape,number_of_input=1):
	ip = Input(shape=input_shape)
	print (input_shape)
	conv1=SeparableConv2D(48, (1, input_shape[1]), activation='relu', padding='valid',kernel_regularizer=regularizers.l1(0.01))(ip)
	conv1=Dropout(0.25)(conv1)
	conv1_bn=BatchNormalization()(conv1)
	conv1_gt1=Convolution2D(48, (2, 1), activation='sigmoid', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv1_bn)
	conv2=Convolution2D(32, (5, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv1_bn)
	conv2=BatchNormalization()(conv2)
	conv3=Convolution2D(32, (5, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv2)
	conv3=BatchNormalization()(conv3)
	conv4=Convolution2D(48, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv3)
	conv1_4=multiply([conv1_bn,conv1_gt1])
	conv5=BatchNormalization()(conv1_4)

	pool1=MaxPooling2D(pool_size=(2,1),strides=(2,1))(conv5)
	pool1=Dropout(0.25)(pool1)
	conv1_gt2=Convolution2D(64, (2, 1), activation='sigmoid', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool1)
	# conv4_branch=Convolution2D(64, (1, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool1)
	conv4=Convolution2D(64, (2, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool1)
	conv4=BatchNormalization()(conv4)
	conv5=Convolution2D(64, (2, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv4)
	# conv2_5=add([conv4_branch,conv5])
	conv2_5=multiply([conv1_gt2,conv5])
	conv_last=BatchNormalization()(conv2_5)
	conv_last=Dropout(0.25)(conv_last)
	# ft=Flatten()(conv7)
	ft =Flatten()(conv_last)
	d1=Dense(128,activation='relu',kernel_regularizer=regularizers.l2(0.01))(ft)
	d1=BatchNormalization()(d1)
	d1=Dropout(0.5)(d1)
	out1 =Dense(1)(d1)

	my_dense =Dense(128,activation='relu',kernel_regularizer=regularizers.l2(0.01))

	d2=my_dense(d1)
	# d2=Dense(128,activation='relu',kernel_regularizer=regularizers.l2(0.01))(d1)
	d2=BatchNormalization()(d2)
	d2=Dropout(0.5)(d2)
	out2 =Dense(1)(d2)

	d3_in=add([d1,d2])
	d3=my_dense(d3_in)
	# d3=Dense(128,activation='relu',kernel_regularizer=regularizers.l2(0.01))(d2)
	d3=BatchNormalization()(d3)
	d3=Dropout(0.5)(d3)
	out3 =Dense(1)(d3)

	d4_in=add([d1,d2,d3])
	d4=my_dense(d4_in)
	# d4=Dense(128,activation='relu',kernel_regularizer=regularizers.l2(0.01))(d4_in)
	d4=BatchNormalization()(d4)
	d4=Dropout(0.5)(d4)
	out4 =Dense(1)(d4)

	d5_in=add([d1,d2,d3,d4])
	d5=my_dense(d5_in)
	# d5=Dense(128,activation='relu',kernel_regularizer=regularizers.l2(0.01))(d5_in)
	d5=BatchNormalization()(d5)
	d5=Dropout(0.5)(d5)
	out5 =Dense(1)(d5)

	d6_in=add([d1,d2,d3,d4,d5])
	d6=my_dense(d6_in)
	# d6=my_dense(128,activation='relu',kernel_regularizer=regularizers.l2(0.01))(d6_in)
	d6=BatchNormalization()(d6)
	d6=Dropout(0.5)(d6)
	out6 =Dense(1)(d6)

	out =concatenate([out1,out2,out3,out4,out5,out6])
	return ip,out
def kdd_model_simple_gated_deep_narrow(input_shape,output_shape,number_of_input=1):
	ip = Input(shape=input_shape)
	print (input_shape)
	conv1=SeparableConv2D(32, (1, input_shape[1]), activation='relu', padding='valid',kernel_regularizer=regularizers.l1(0.01))(ip)
	# conv1=Dropout(0.5)(conv1)
	conv1_bn=BatchNormalization()(conv1)
	conv1_gt1=Convolution2D(32, (2, 1), activation='sigmoid', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv1_bn)
	conv2=Convolution2D(16, (5, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv1_bn)
	conv2=BatchNormalization()(conv2)
	conv3=Convolution2D(16, (5, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv2)
	conv3=BatchNormalization()(conv3)
	conv4=Convolution2D(16, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv3)
	conv4=BatchNormalization()(conv4)

	conv5=Convolution2D(16, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv4)
	conv5=BatchNormalization()(conv5)
	conv6=Convolution2D(32, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv5)
	conv1_4=multiply([conv6,conv1_gt1])
	conv1_4=BatchNormalization()(conv1_4)

	pool1=MaxPooling2D(pool_size=(2,1),strides=(2,1))(conv1_4)
	pool1=Dropout(0.25)(pool1)
	conv1_gt2=Convolution2D(24, (2, 1), activation='sigmoid', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool1)
	# conv4_branch=Convolution2D(64, (1, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool1)
	conv6=Convolution2D(24, (2, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool1)
	conv6=BatchNormalization()(conv6)
	conv7=Convolution2D(24, (2, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv6)
	conv7=BatchNormalization()(conv7)
	conv8=Convolution2D(24, (2, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv7)
	conv8=BatchNormalization()(conv8)
	conv9=Convolution2D(24, (2, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv8)
	# conv2_5=add([conv4_branch,conv5])
	conv2_9=multiply([conv1_gt2,conv9])
	conv_last=BatchNormalization()(conv2_9)
	conv_last=Dropout(0.25)(conv_last)
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
def kdd_model_simple_gated_conv_LSMT(input_shape,output_shape,number_of_input=1):
	ip = Input(shape=input_shape)
	print (input_shape)
	conv1=SeparableConv2D(48, (1, input_shape[1]), activation='relu', padding='valid',kernel_regularizer=regularizers.l1(0.01))(ip)
	# conv1=Dropout(0.5)(conv1)
	conv1_bn=BatchNormalization()(conv1)
	conv1_gt1=Convolution2D(48, (2, 1), activation='sigmoid', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv1_bn)
	conv2=Convolution2D(32, (5, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv1_bn)
	conv2=BatchNormalization()(conv2)
	conv3=Convolution2D(32, (5, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv2)
	conv3=BatchNormalization()(conv3)
	conv4=Convolution2D(48, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv3)
	# conv1_4=multiply([conv1_bn,conv1_gt1])
	conv1_4=multiply([conv4,conv1_gt1])
	conv5=BatchNormalization()(conv1_4)

	pool1=MaxPooling2D(pool_size=(2,1),strides=(2,1))(conv5)
	pool1=Dropout(0.25)(pool1)
	conv1_gt2=Convolution2D(64, (2, 1), activation='sigmoid', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool1)
	# conv4_branch=Convolution2D(64, (1, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool1)
	conv4=Convolution2D(64, (2, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool1)
	conv4=BatchNormalization()(conv4)
	conv5=Convolution2D(64, (2, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv4)
	# conv2_5=add([conv4_branch,conv5])
	conv2_5=multiply([conv1_gt2,conv5])
	conv_last=BatchNormalization()(conv2_5)
	conv_last=Dropout(0.25)(conv_last)
	time_data=Reshape((-1,64))(conv_last)
	x1=LSTM(48,return_sequences=True)(time_data)
	x=LSTM(32,return_sequences=False)(x1)
	# ft=Flatten()(conv
	# ft =Flatten()(conv_last)
	out=Dense(6)(x)
	# d=BatchNormalization()(d)
	# d=Dropout(0.5)(d)
	# print(d)
	# n =output_shape[0]
	# print(n)
	# out =Dense(n)(d)
	return ip,out
def kdd_model_simple_gated(input_shape,output_shape,number_of_input=1):
	ip = Input(shape=input_shape)
	print (input_shape)
	conv1=SeparableConv2D(48, (1, input_shape[1]), activation='relu', padding='valid',kernel_regularizer=regularizers.l1(0.01))(ip)
	conv1=Dropout(0.25)(conv1)
	conv1_bn=BatchNormalization()(conv1)
	conv1_gt1=Convolution2D(48, (2, 1), activation='sigmoid', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv1_bn)
	conv2=Convolution2D(32, (5, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv1_bn)
	conv2=BatchNormalization()(conv2)
	conv3=Convolution2D(32, (5, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv2)
	conv3=BatchNormalization()(conv3)
	conv4=Convolution2D(48, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv3)
	# conv1_4=multiply([conv1_bn,conv1_gt1])
	conv1_4=multiply([conv4,conv1_gt1])
	conv5=BatchNormalization()(conv1_4)

	pool1=MaxPooling2D(pool_size=(2,1),strides=(2,1))(conv5)
	pool1=Dropout(0.25)(pool1)
	conv1_gt2=Convolution2D(64, (2, 1), activation='sigmoid', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool1)
	# conv4_branch=Convolution2D(64, (1, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool1)
	conv4=Convolution2D(64, (2, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool1)
	conv4=BatchNormalization()(conv4)
	conv5=Convolution2D(64, (2, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv4)
	# conv2_5=add([conv4_branch,conv5])
	conv2_5=multiply([conv1_gt2,conv5])
	conv_last=BatchNormalization()(conv2_5)
	conv_last=Dropout(0.25)(conv_last)
	# ft=Flatten()(conv7)
	ft =Flatten()(conv_last)
	d=Dense(64)(ft)
	d=BatchNormalization()(d)
	d=Dropout(0.5)(d)
	# print(d)
	n =output_shape[0]
	print(n)
	out =Dense(n)(d)
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
	pool1=Dropout(0.25)(pool1)
	conv4_branch=Convolution2D(64, (1, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool1)
	conv4=Convolution2D(64, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool1)
	conv4=BatchNormalization()(conv4)
	conv5=Convolution2D(64, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv4)
	conv2_5=add([conv4_branch,conv5])
	conv_last=BatchNormalization()(conv2_5)
	conv_last=Dropout(0.25)(conv_last)
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
	pool1=Dropout(0.25)(pool1)
	gate_conv3 =Convolution2D(128, (3, 1), activation='sigmoid', padding='same')(pool1)
	conv4=Convolution2D(128, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool1)
	conv4=BatchNormalization()(conv4)
	conv5=Convolution2D(128, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv4)
	conv5=BatchNormalization()(conv5)
	gate_conv3 = multiply([gate_conv3,conv5])
	pool2=MaxPooling2D(pool_size=(3,1),strides=(2,1))(gate_conv3 )
	pool2=Dropout(0.25)(pool2)

	gate_conv4=Convolution2D(256, (3, 1), activation='sigmoid', padding='same')(pool2)
	conv6=Convolution2D(256, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool2)
	conv6=BatchNormalization()(conv6)
	conv7=Convolution2D(256, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv6)
	conv7=BatchNormalization()(conv7)
	gate_conv4 = multiply([gate_conv4,conv7])
	gate_conv4=Dropout(0.25)(gate_conv4)

	
	ft=Flatten()(gate_conv4)
	d=Dense(382,kernel_regularizer=regularizers.l2(0.01))(ft)
	d=BatchNormalization()(d)
	d=Dropout(0.5)(d)
	# print(d)
	n =output_shape[0]
	print(n)
	out =Dense(n)(d)
	return ip,out
def kdd_gated_conv_LSTM(input_shape,output_shape,number_of_input=1):
	ip = Input(shape=input_shape)
	print (input_shape)
	#em=Embedding(100,100)(ip)
	conv0=SeparableConv2D(128, (1, input_shape[1]), activation='relu', padding='valid',use_bias=False,kernel_regularizer=regularizers.l1(0.01))(ip)
	conv0=Dropout(0.5)(conv0)
	conv0=BatchNormalization()(conv0)
	embeded_tensor=Permute((1,3,2))(conv0)
	conv1=Convolution2D(64, (1, 128), activation='relu', padding='valid',use_bias=False,kernel_regularizer=regularizers.l2(0.01))(embeded_tensor)

	# conv1=Dropout(0.5)(conv1)
	conv1=BatchNormalization()(conv1)
	gate_conv1 =Convolution2D(48, (5, 1), activation='sigmoid', padding='same')(conv1)
	conv2=Convolution2D(48, (5, 1), activation='relu', padding='same',use_bias=False,kernel_regularizer=regularizers.l2(0.01))(conv1)
	conv2=BatchNormalization()(conv2)
	conv3=Convolution2D(48, (5, 1), activation='relu', padding='same',use_bias=False,kernel_regularizer=regularizers.l2(0.01))(conv2)
	gate_conv2 = multiply([gate_conv1,conv3])
	gate_conv2=BatchNormalization()(gate_conv2)

	pool1=MaxPooling2D(pool_size=(2,1),strides=(2,1))(gate_conv2)
	pool1=Dropout(0.25)(pool1)
	gate_conv3 =Convolution2D(64, (5, 1), activation='sigmoid', padding='same')(pool1)
	conv4=Convolution2D(64, (5, 1), activation='relu', padding='same',use_bias=False,kernel_regularizer=regularizers.l2(0.01))(pool1)
	conv4=BatchNormalization()(conv4)
	conv5=Convolution2D(64, (5, 1), activation='relu', padding='same',use_bias=False,kernel_regularizer=regularizers.l2(0.01))(conv4)
	# conv5=BatchNormalization()(conv5)
	gate_conv3 = multiply([gate_conv3,conv5])
	gate_conv3=BatchNormalization()(gate_conv3)
	pool2=MaxPooling2D(pool_size=(2,1),strides=(2,1))(gate_conv3 )
	pool2=Dropout(0.25)(pool2)

	gate_conv4=Convolution2D(128, (3, 1), activation='sigmoid', padding='same')(pool2)
	conv6=Convolution2D(128, (3, 1), activation='relu', padding='same',use_bias=False,kernel_regularizer=regularizers.l2(0.01))(pool2)
	conv6=BatchNormalization()(conv6)
	conv7=Convolution2D(128, (3, 1), activation='relu', padding='same',use_bias=False,kernel_regularizer=regularizers.l2(0.01))(conv6)
	# conv7=BatchNormalization()(conv7)
	gate_conv4 = multiply([gate_conv4,conv7])
	gate_conv4=BatchNormalization()(gate_conv4)
	time_data=Reshape((-1,128))(gate_conv4)

	x=LSTM(64,return_sequences=False)(time_data)
	# x=LSTM(32,return_sequences=False)(x1)
	# ft=Flatten()(conv
	# ft =Flatten()(conv_last)
	out=Dense(6)(x)
	return ip,out

	
	# ft=Flatten()(gate_conv4)
	# d=Dense(128,kernel_regularizer=regularizers.l2(0.01))(ft)
	# d=BatchNormalization()(d)
	# d=Dropout(0.5)(d)
	# # print(d)
	# n =output_shape[0]
	# print(n)
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
	gate_conv2 = multiply([gate_conv1,conv3])
	gate_conv2=BatchNormalization()(gate_conv2)

	pool1=MaxPooling2D(pool_size=(2,1),strides=(2,1))(gate_conv2)
	pool1=Dropout(0.25)(pool1)
	gate_conv3 =Convolution2D(64, (5, 1), activation='sigmoid', padding='same')(pool1)
	conv4=Convolution2D(64, (5, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool1)
	conv4=BatchNormalization()(conv4)
	conv5=Convolution2D(64, (5, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv4)
	# conv5=BatchNormalization()(conv5)
	gate_conv3 = multiply([gate_conv3,conv5])
	gate_conv3=BatchNormalization()(gate_conv3)
	pool2=MaxPooling2D(pool_size=(2,1),strides=(2,1))(gate_conv3 )
	pool2=Dropout(0.25)(pool2)

	gate_conv4=Convolution2D(128, (3, 1), activation='sigmoid', padding='same')(pool2)
	conv6=Convolution2D(128, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool2)
	conv6=BatchNormalization()(conv6)
	conv7=Convolution2D(128, (3, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv6)
	# conv7=BatchNormalization()(conv7)
	gate_conv4 = multiply([gate_conv4,conv7])
	gate_conv4=BatchNormalization()(gate_conv4)

	
	ft=Flatten()(gate_conv4)
	d=Dense(128,kernel_regularizer=regularizers.l2(0.01))(ft)
	d=BatchNormalization()(d)
	d=Dropout(0.5)(d)
	# print(d)
	n =output_shape[0]
	print(n)
	out =Dense(n)(d)
	return ip,out

def kdd_gated_model_3blocks(input_shape,output_shape,number_of_input=1):
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
	gate_conv2 = multiply([gate_conv1,conv3])
	gate_conv2=BatchNormalization()(gate_conv2)

	pool1=MaxPooling2D(pool_size=(2,1),strides=(2,1))(gate_conv2)
	pool1=Dropout(0.25)(pool1)
	gate_conv3 =Convolution2D(64, (5, 1), activation='sigmoid', padding='same')(pool1)
	conv4=Convolution2D(64, (5, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool1)
	conv4=BatchNormalization()(conv4)
	conv5=Convolution2D(64, (5, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv4)
	# conv5=BatchNormalization()(conv5)
	gate_conv3 = multiply([gate_conv3,conv5])
	gate_conv3=BatchNormalization()(gate_conv3)
	pool2=MaxPooling2D(pool_size=(2,1),strides=(2,1))(gate_conv3 )
	pool2=Dropout(0.25)(pool2)

	gate_conv4=Convolution2D(128, (5, 1), activation='sigmoid', padding='same')(pool2)
	conv6=Convolution2D(128, (5, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool2)
	conv6=BatchNormalization()(conv6)
	conv7=Convolution2D(128(5, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv6)
	# conv7=BatchNormalization()(conv7)
	gate_conv4 = multiply([gate_conv4,conv7])
	gate_conv4=BatchNormalization()(gate_conv4)
	pool3=MaxPooling2D(pool_size=(2,1),strides=(2,1))(gate_conv4 )
	pool3=Dropout(0.25)(pool3)

	gate_conv5=Convolution2D(196, (5, 1), activation='sigmoid', padding='same')(pool3)
	conv8=Convolution2D(196, (5, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(pool3)
	conv8=BatchNormalization()(conv8)
	conv9=Convolution2D(196, (5, 1), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.01))(conv8)
	# conv7=BatchNormalization()(conv7)
	gate_conv5 = multiply([gate_conv5,conv9])
	gate_conv5=BatchNormalization()(gate_conv5)

	
	ft=Flatten()(gate_conv5)
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
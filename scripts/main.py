import kdd_data
import kdd_deep_models
import numpy as np
from keras.layers import Input
from keras.callbacks  import CSVLogger, ModelCheckpoint, TensorBoard
from keras.models import Model
from keras import backend as K
from keras.optimizers import RMSprop, SGD,Adam
import matplotlib.pyplot as plt
from keras import metrics
import os

def train_deep_model(X_train,Y_train,model_file_prefix=None):
	# row =75
	# col =24
	n,h,w,c =X_train.shape
	# ip = Input(shape=(h, w,c))
	input_shape =(h, w,c)
	if model_file_prefix is not None:
		model_name=model_file_prefix
	else:
		model_name ='sample_conv_nosuffle'
		# model_name ='sample_conv'
	n,out_n =Y_train.shape
	output_shape=(out_n,)
	ip,out=kdd_deep_models.kdd_model(input_shape,output_shape)
	model=Model(ip,out)

	weight_h5_file='./'+ model_name +'.h5'
	if os.path.isfile(weight_h5_file):
		try:
			model.load_weights(weight_h5_file)
		except:
			print ('the model {} can not  be loaded'.format(weight_h5_file))
			pass
	# from sklearn.model_selection import KFold
	# f = KFold(n_splits=2)
	# optimizer =RMSprop(lr = 1e-4)
	optimizer =Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)
	model.compile(optimizer=optimizer, loss='mean_absolute_percentage_error', metrics=[metrics.mape])
	model.summary()
	best_model = ModelCheckpoint(weight_h5_file, verbose = 1, save_best_only = True)
	tensorboard= TensorBoard(log_dir='./logs',histogram_freq=0,write_graph=True,write_images=False)
	history=model.fit(x=X_train,y=Y_train, epochs = 500, validation_split=0.2, callbacks = [tensorboard,best_model])
def model_predict(X_test,output_shape,model_file_prefix=None):
	n,h,w,c =X_test.shape
	# ip = Input(shape=(h, w,c))
	input_shape =(h, w,c)
	if model_file_prefix is not None:
		model_name=model_file_prefix
	else:
		model_name ='sample_conv_nosuffle'
	# model_name ='sample_conv'
	weight_h5_file='./'+ model_name +'.h5'
	ip,out=kdd_deep_models.kdd_model(input_shape,output_shape)
	model=Model(ip,out)
	if os.path.isfile(weight_h5_file):
		try:
			model.load_weights(weight_h5_file)
		except:
			print ('the model {} can not  be loaded'.format(weight_h5_file))
			return
	return model.predict(X_test)
	# optimizer =RMSprop(lr = 1e-4)
	# model.compile(optimizer=optimizer, loss='mean_absolute_percentage_error', metrics=['mape'])
# def mape(y_true,y_pred):
	# return K.abs(y_true-y_pred)/K.sum(y_true)	
def train_cross_validation_models():
	pass
if __name__ == "__main__":
	kdd_DATA =kdd_data.kdd_data()

	# route_time_windows = list(kdd_DATA.travel_times['C-3'].keys())
	# route_time_windows.sort()
	# print route_time_windows[0:60]
	from sklearn.metrics import mean_absolute_error
	from sklearn.metrics import explained_variance_score
	clf={}
	# for route_id in kdd_DATA.travel_times:
	# route_id='B-3'
	# # kdd_DATA.travel_times=kdd_DATA.zerofill_missed_time_info(kdd_DATA.travel_times,route_id)
	# mat =kdd_DATA.get_feature_matrix(kdd_DATA.travel_times,route_id)
	# X_train,Y_train =kdd_DATA.prepare_train_data(mat)

	##-------------------- concatenate all routes information and train & predict in one model----- 
	# X_train_list =[]
	# Y_train_list=[]
	# kdd_DATA.travel_times=kdd_DATA.zerofill_missed_time_info(kdd_DATA.travel_times)
	# for route_id in kdd_DATA.travel_times:
	# 	mat =kdd_DATA.get_feature_matrix(kdd_DATA.travel_times,route_id)
	# 	X_train_c,Y_train_c =kdd_DATA.prepare_train_data(mat)
	# 	X_train_list.append(np.expand_dims(X_train_c,axis=2))
	# 	Y_train_list.append(Y_train_c)
	# X_train=np.concatenate(X_train_list,axis=2)
	# Y_train=np.concatenate(Y_train_list,axis=1)
	# n,d,c=X_train.shape
	# time_d =int(2*60/kdd_DATA.time_interval)
	# X_train_2D = np.reshape(X_train,(n,time_d,-1,c))

	#----------------------------------------------------------------------------------------

	#------------------------- train single route ------------------------------------

	route_id='B-3'
	# travel_time_test_info=kdd_data.read_test_data()
	# mat_test=kdd_data.get_test_feature_mat(travel_time_test_info,route_id)
	# import ipdb
	# ipdb.set_trace()

	kdd_DATA.travel_times=kdd_DATA.zerofill_missed_time_info(kdd_DATA.travel_times,route_id)
	mat =kdd_DATA.get_feature_matrix(kdd_DATA.travel_times,route_id)
	X_train,Y_train =kdd_DATA.prepare_train_data(mat)
	n,d=X_train.shape
	time_d =int(2*60/kdd_DATA.time_interval)
	saved_model_file_name =route_id +'time_estimate'
	X_train_2D = np.reshape(X_train,(n,time_d,-1,1))
	

	##---------------------------------------------------------------------------------------##

	# X_train=np.concatenate((X_train1,X_train2[:6540]),aixs=1)
	
	
	from sklearn.utils import shuffle
	# X_train_2D,Y_train=shuffle(X_train_2D,Y_train,random_state=0)
	train_deep_model(X_train_2D,Y_train,saved_model_file_name)

	n,outp=Y_train.shape
	outshape =(outp,)
	Y_p=model_predict(X_train_2D,outshape,saved_model_file_name)
	print("mape = {}".format(mean_absolute_error(Y_train,Y_p)))
	print("variance score = {}".format(explained_variance_score(Y_train,Y_p)))



	# clf= linear_model.MultiTaskLasso(alpha=0.1,max_iter=10000)
	# clf.fit(X_train,Y_train)
	# Y_p =	clf.predict(X_train)
	# print("mean erros of route {}".format(route_id))
	# print(mean_absolute_error(Y_train,Y_p))
	# print(explained_variance_score(Y_train,Y_p))
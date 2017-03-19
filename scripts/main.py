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

def train_deep_model(X_train,Y_train):
	# row =75
	# col =24
	n,h,w,c =X_train.shape
	# ip = Input(shape=(h, w,c))
	input_shape =(h, w,c)
	model_name ='sample_conv'
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


	# optimizer =RMSprop(lr = 1e-4)
	optimizer =Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)
	model.compile(optimizer=optimizer, loss='mean_absolute_percentage_error', metrics=[metrics.mape])
	model.summary()
	best_model = ModelCheckpoint(weight_h5_file, verbose = 1, save_best_only = True)
	tensorboard= TensorBoard(log_dir='./logs',histogram_freq=0,write_graph=True,write_images=False)
	history=model.fit(x=X_train,y=Y_train, epochs = 500, validation_split=0.2, callbacks = [tensorboard,best_model])
def model_predict(X_test):
	n,h,w,c =X_test.shape
	# ip = Input(shape=(h, w,c))
	shape =(h, w,c)
	model_name ='sample_conv'
	weight_h5_file='./'+ model_name +'.h5'
	ip,out=kdd_deep_models.kdd_model(shape)
	model=Model(ip,out)
	model.load_weights(weight_h5_file)
	return model.predict(X_test)
	# optimizer =RMSprop(lr = 1e-4)
	# model.compile(optimizer=optimizer, loss='mean_absolute_percentage_error', metrics=['mape'])
	

# def mape(y_true,y_pred):
	# return K.abs(y_true-y_pred)/K.sum(y_true)
if __name__ == "__main__":
	A =kdd_data.kdd_data()

	# route_time_windows = list(A.travel_times['C-3'].keys())
	# route_time_windows.sort()
	# print route_time_windows[0:60]
	from sklearn.metrics import mean_absolute_error
	from sklearn.metrics import explained_variance_score
	clf={}
	# for route_id in A.travel_times:
	# route_id='B-3'
	# A.travel_times=A.zerofill_missed_time_info(A.travel_times,route_id)
	# mat =A.get_feature_matrix(A.travel_times,route_id)
	# X_train1,Y_train =A.prepare_train_data(mat)


	route_id='A-3'
	# A.travel_times=A.zerofill_missed_time_info(A.travel_times,route_id)
	mat =A.get_feature_matrix(A.travel_times,route_id)
	X_train2,Y_train =A.prepare_train_data(mat)

	# X_train=np.concatenate((X_train1,X_train2[:6540]),aixs=1)
	n,d=X_train2.shape
	time_d =int(2*60/A.time_interval)
	X_train_2D = np.reshape(X_train2,(n,time_d,-1,1))
	from sklearn.utils import shuffle
	X_train_2D,Y_train=shuffle(X_train_2D,Y_train,random_state=0)
	train_deep_model(X_train_2D,Y_train)



	# clf= linear_model.MultiTaskLasso(alpha=0.1,max_iter=10000)
	# clf.fit(X_train,Y_train)
	# Y_p =	clf.predict(X_train)
	# print("mean erros of route {}".format(route_id))
	# print(mean_absolute_error(Y_train,Y_p))
	# print(explained_variance_score(Y_train,Y_p))
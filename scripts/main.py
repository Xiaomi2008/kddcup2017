import kdd_data
import kdd_deep_models
import numpy as np
from keras.layers import Input
from keras.callbacks  import CSVLogger, ModelCheckpoint, TensorBoard
from keras.models import Model
from keras import backend as K
from keras.optimizers import RMSprop, SGD
import matplotlib.pyplot as plt

def train_deep_model(X_train,Y_train):
	# row =75
	# col =24
	n,h,w,c =X_train.shape
	# ip = Input(shape=(h, w,c))
	shape =(h, w,c)
	model_name ='sample_conv'
	ip,out=kdd_deep_models.kdd_model(shape)
	model=Model(ip,out)
	optimizer =RMSprop(lr = 1e-4)
	model.compile(optimizer=optimizer, loss='mean_absolute_percentage_error', metrics=['mape'])
	weight_h5_file='./'+ model_name +'.h5'
	best_model = ModelCheckpoint(weight_h5_file, verbose = 1, save_best_only = True)
	tensorboard= TensorBoard(log_dir='./logs',histogram_freq=0,write_graph=True,write_images=False)
	history=model.fit(x=X_train,y=Y_train, epochs = 500, validation_split=0.2, callbacks = [tensorboard,best_model])
def mape(y_true,y_pred):
	return K.abs(y_true-y_pred)/K.sum(y_true)
if __name__ == "__main__":
	A =kdd_data.kdd_data()

	# route_time_windows = list(A.travel_times['C-3'].keys())
	# route_time_windows.sort()
	# print route_time_windows[0:60]
	from sklearn.metrics import mean_absolute_error
	from sklearn.metrics import explained_variance_score
	clf={}
	# for route_id in A.travel_times:
	route_id='B-3'
	# A.travel_times=A.zerofill_missed_time_info(A.travel_times,route_id)
	mat =A.get_feature_matrix(A.travel_times,route_id)
	X_train,Y_train =A.prepare_train_data(mat)
	n,d=X_train.shape
	time_d =int(2*60/A.time_interval)
	X_train_2D = np.reshape(X_train,(n,time_d,-1,1))

	# from sklearn.utils import shuffle
	# X_train_2D,Y_train=shuffle(X_train_2D,Y_train,random_state=0)

	train_deep_model(X_train_2D,Y_train)



	# clf[route_id] = linear_model.MultiTaskLasso(alpha=0.1,max_iter=20000)
	# clf[route_id].fit(X_train,Y_train)
	# Y_p =	clf[route_id].predict(X_train)
	# print("mean erros of route {}".format(route_id))
	# print(mean_absolute_error(Y_train,Y_p))
	# print(explained_variance_score(Y_train,Y_p))
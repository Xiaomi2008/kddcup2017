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
from datetime import datetime,timedelta
import os
import argparse


time_range_day1_am =('2016-10-18 06:00:00','2016-10-18 08:00:00')
time_range_day1_pm =('2016-10-18 15:00:00','2016-10-18 17:00:00')

time_range_day2_am =('2016-10-19 06:00:00','2016-10-19 08:00:00')
time_range_day2_pm =('2016-10-19 15:00:00','2016-10-19 17:00:00')

time_range_day3_am =('2016-10-20 06:00:00','2016-10-20 08:00:00')
time_range_day3_pm =('2016-10-20 15:00:00','2016-10-20 17:00:00')

time_range_day4_am =('2016-10-21 06:00:00','2016-10-21 08:00:00')
time_range_day4_pm =('2016-10-21 15:00:00','2016-10-21 17:00:00')

time_range_day5_am =('2016-10-22 06:00:00','2016-10-22 08:00:00')
time_range_day5_pm =('2016-10-22 15:00:00','2016-10-22 17:00:00')

time_range_day6_am =('2016-10-23 06:00:00','2016-10-23 08:00:00')
time_range_day6_pm =('2016-10-23 15:00:00','2016-10-23 17:00:00')

time_range_day7_am =('2016-10-24 06:00:00','2016-10-24 08:00:00')
time_range_day7_pm =('2016-10-24 15:00:00','2016-10-24 17:00:00')

test_time_list =[time_range_day1_am,time_range_day1_pm,time_range_day2_am,time_range_day2_pm, \
					time_range_day3_am,time_range_day3_pm,time_range_day4_am,time_range_day4_pm, \
					time_range_day5_am,time_range_day5_pm,time_range_day6_am,time_range_day6_pm, \
					time_range_day7_am,time_range_day7_pm]

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
	history=model.fit(x=X_train,y=Y_train, epochs = 100, validation_split=0.05, callbacks = [tensorboard,best_model])
def model_predict(X_test,output_shape,model_file_prefix=None):
	global previous_model_file
	global previous_model
	n,h,w,c =X_test.shape
	# ip = Input(shape=(h, w,c))
	input_shape =(h, w,c)
	if model_file_prefix is not None:
		model_name=model_file_prefix
	else:
		model_name ='sample_conv_nosuffle'
	# model_name ='sample_conv'
	weight_h5_file='./'+ model_name +'.h5'
	if previous_model_file is None or previous_model_file != model_name:
		print("model predicting ....")
		ip,out=kdd_deep_models.kdd_model(input_shape,output_shape)
		model=Model(ip,out)
		previous_model_file =model_file_prefix
		previous_model =model
		print("model predicting ...." +weight_h5_file)
		model.load_weights(weight_h5_file)
		# if os.path.isfile(weight_h5_file):
		# 	try:
		# 		# import ipdb
		# 		# ipdb.set_trace()
		# 		print("loading model file :   "+weight_h5_file)
		# 		model.load_weights(weight_h5_file)
		# 	except:
		# 		print ('the model {} can not  be loaded'.format(weight_h5_file))
		# 		return
	else:
		model = previous_model
	return model.predict(X_test)
	# optimizer =RMSprop(lr = 1e-4)
	# model.compile(optimizer=optimizer, loss='mean_absolute_percentage_error', metrics=['mape'])
# def mape(y_true,y_pred):
	# return K.abs(y_true-y_pred)/K.sum(y_true)	
def train_cross_validation_models():
	pass
def write_submit_traj_file(predict_result):
	out_file_name ='predict_20min_avg_travel_time.csv'
	fw = open(out_file_name, 'w')
	fw.writelines(','.join(['"intersection_id"', '"tollgate_id"', '"time_window"', '"avg_travel_time"']) + '\n')
	for route_id in predict_result.keys():
		route_time_windows = list(predict_result[route_id].keys())
		route_time_windows.sort()
		for time_window_start in route_time_windows:
			p_tts=predict_result[route_id][time_window_start]
			p_start_time = time_window_start +timedelta(hours=2)
			p_end_time = p_start_time + timedelta(minutes=20)
			p_ave_time = p_tts[0,0]
			for i in range(6):
				out_line=','.join(['"' + route_id.split('-')[0] + '"', '"' + route_id.split('-')[1] + '"',
                                 '"[' + str(p_start_time) + ',' + str(p_end_time) + ')"',
                                 '"' + str(p_ave_time) + '"']) + '\n'
				fw.writelines(out_line)
				p_start_time += timedelta(minutes=20)
				p_end_time = p_start_time + timedelta(minutes=20)
				if i<5:
					p_ave_time = p_tts[0,i+1]
	fw.close()
if __name__ == "__main__":
	parser =argparse.ArgumentParser()
	parser.add_argument("-p","--phase", help="train of test",default='train')
	parser.add_argument("-r","--route_id", help="given a route_id for model to train",default='B-3')
	parser.add_argument("-i","--interval", help="feature inverval minutes",type=int,default=5)
	args=parser.parse_args()
	phase =args.phase
	kdd_DATA =kdd_data.kdd_data(args.interval)
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
	global previous_model_file
	global previous_model
	previous_model_file =None
	previous_model =None

	#----------------------------------------------------------------------------------------

	#------------------------- test all routes ------------------------------------
	
	route_ids=['A-2','A-3','C-1','C-3','B-1','B-3']

	# route_ids=['A-2']
	if phase =='test':
		# route_id='A-3'
		
		travel_time_test_info=kdd_DATA.read_test_data()
		predict_result ={}
		for route_id in route_ids:
			predict_result[route_id]={}
			for tt in test_time_list:
				start_time=datetime.strptime(tt[0],"%Y-%m-%d %H:%M:%S")
				end_time=datetime.strptime(tt[1],"%Y-%m-%d %H:%M:%S")
				travel_time_test_info=kdd_DATA.zerofill_missed_time_info(travel_time_test_info,route_id,
																	start_time,end_time)
			mat_test=kdd_DATA.get_test_feature_mat(travel_time_test_info,route_id)
			time_list =list(mat_test.keys())
			time_list.sort()
			saved_model_file_name =route_id +'_time_estimate_'+ 'model_2_'+str(kdd_DATA.time_interval)
			# saved_model_file_name =route_id +'time_estimate' #+str(kdd_DATA.time_interval)	
			for i in range(len(time_list)):
				test_X = np.array(mat_test[time_list[i]])
				test_X=np.expand_dims(test_X,axis=2)
				test_X=np.expand_dims(test_X,axis=0)
				num_out_put =[6]
				predict_result[route_id][time_list[i]]=model_predict(test_X,num_out_put,saved_model_file_name)
		write_submit_traj_file(predict_result)

	# import ipdb
	# ipdb.set_trace()
	if phase =='train':
		# route_id='A-2'
		route_id =args.route_id
		# kdd_DATA.travel_times=kdd_DATA.format_data_in_timeInterval(traj_data,vol_data)
		kdd_DATA.read_train_data()
		kdd_DATA.travel_times=kdd_DATA.zerofill_missed_time_info(travel_times=kdd_DATA.travel_times,route_id=route_id)
		mat =kdd_DATA.get_feature_matrix(kdd_DATA.travel_times,route_id)
		X_train,Y_train =kdd_DATA.prepare_train_data(mat)
		n,d=X_train.shape
		time_d =int(2*60/kdd_DATA.time_interval)
		saved_model_file_name =route_id +'_time_estimate_' +'model_2_'+ str(kdd_DATA.time_interval)
		X_train_2D = np.reshape(X_train,(n,time_d,-1,1))
		
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
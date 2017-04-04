import kdd_data
import kdd_deep_models
import numpy as np
from numpy import random
from keras.layers import Input
from keras.callbacks  import CSVLogger, ModelCheckpoint, TensorBoard
from keras.models import Model
from keras import backend as K
from keras.optimizers import RMSprop, SGD,Adam
import matplotlib.pyplot as plt
from keras import metrics
from keras import losses
from datetime import datetime,timedelta
import os
import argparse
import lightgbm as lgb
import h5py
import ipdb

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

deep_models={}
deep_models['gated_cnn_1']=kdd_deep_models.kdd_gated_model
deep_models['simple_inception']=kdd_deep_models.kdd_model_inception
deep_models['simple_gate']=kdd_deep_models.kdd_model_simple_gated
deep_models['simple_gate_condition']=kdd_deep_models.kdd_model_simple_gated_condition_prob
deep_models['simple_gate_deep_narrow']=kdd_deep_models.kdd_model_simple_gated_deep_narrow
deep_models['LSTM_1']=kdd_deep_models.kdd_LSTM_1
deep_models['LSTM_2']=kdd_deep_models.kdd_LSTM_2
deep_models['LSTM_2_bidirection']=kdd_deep_models.kdd_LSTM_bidirection
deep_models['default_model']=deep_models['gated_cnn_1']
deep_models['GBM']='GBM'
# deep_models['gated_cnn']=kdd_deep_models.kdd_gated_model
def get_model_and_weightFile(args):
	model = deep_models[args.model]
	# print (args)
	phase =args.phase
	model_save_file ='time_'+args.model +'_'+args.route_id +'_' +'I'+str(args.interval) +'_T'+str(args.given_time)
	combined_routeID_features =''
	if hasattr(args,'clist'):
		for r_id in args.clist:
			combined_routeID_features='-'+combined_routeID_features +r_id
		combined_routeID_features='_CombineF'+'__'+combined_routeID_features +'__'
	model_save_file =model_save_file +combined_routeID_features+'.h5'
	return model, model_save_file
def change_sample_weight(sample_weights,d1,d2):
	max_r=np.max(sample_weights)
	idx=sample_weights==max_r
	sample_weights[idx]=d1
	sample_weights[~idx]=d2
	# ipdb.set_trace()
	return sample_weights

def load_X_Y(args):
	data_file ='X_Y_'+args.route_id +'_' +'I'+str(args.interval) +'_T'+str(args.given_time)
	combined_routeID_features =''
	if hasattr(args,'clist'):
		for r_id in args.clist:
			combined_routeID_features='-'+combined_routeID_features +r_id
		combined_routeID_features='_CombineF'+'__'+combined_routeID_features +'__'
	data_file ='./data/'+data_file +combined_routeID_features+'.h5'
	print('load training data from : {} '.format(data_file))
	if os.path.exists(data_file):
		f=h5py.File(data_file,'r')
		X_f=f['X']
		Y_f=f['Y']
		SW_f=f['SW']
		X=X_f[:]
		Y=Y_f[:]
		SW=SW_f[:]
		f.close()
		return X,Y,SW
	else:
		print('Can not find the data file')
		return None,None,None
def save_X_Y(args,X,Y,sample_weights):
	data_file ='X_Y_'+args.route_id +'_' +'I'+str(args.interval) +'_T'+str(args.given_time)
	combined_routeID_features =''
	if hasattr(args,'clist'):
		for r_id in args.clist:
			combined_routeID_features='-'+combined_routeID_features +r_id
		combined_routeID_features='_CombineF'+'__'+combined_routeID_features +'__'
	data_file ='./data/'+data_file +combined_routeID_features+'.h5'
	# if os.path.exists(data_file):
	f=h5py.File(data_file,'w')
	f.create_dataset("./X", data=X)
	f.create_dataset("./Y",data=Y)
	f.create_dataset("./SW",data=sample_weights)
	f.close()
		# return X,Y
def generate_train_data(args):
	X,Y,sample_weights=load_X_Y(args)
	if X is None:
		route_id =args.route_id
			# kdd_DATA.travel_times=kdd_DATA.format_data_in_timeInterval(traj_data,vol_data)
		kdd_DATA.read_train_data()
		X_train_list =[]
		Y_train_list=[]
		cf_list =combined_feature_list[:]
		if route_id not in cf_list:
			cf_list.append(route_id)
		kdd_DATA.travel_times=kdd_DATA.zerofill_missed_time_info(travel_times=kdd_DATA.travel_times,
																		input_route_ids=[route_id],
																		fill_route_ids=cf_list,
																		phase='train')
		input_feature_index =0
		for i,r_id in enumerate(cf_list):
			mat,time_stamp =kdd_DATA.get_feature_matrix(kdd_DATA.travel_times,r_id,kdd_DATA.weather_train)
			X_train_c,Y_train_c,sample_weights =kdd_DATA.prepare_train_data(mat,time_stamp=time_stamp,p_hour=args.given_time)
			X_train_list.append(np.expand_dims(X_train_c,axis=2))
			Y_train_list.append(Y_train_c)
			if r_id ==route_id:
				input_feature_index =i
				Y_train = Y_train_c
		sample_weights=np.array(sample_weights)
		X_train_list,Y_train_list,del_idx=kdd_DATA.delete_zero_y_data(X_train_list,[Y_train],input_feature_index)
		sample_weights=np.delete(sample_weights,del_idx)
		Y_train =Y_train_list[0]
		X_train=np.concatenate(X_train_list,axis=2)
			

			# Y_train=np.concatenate(Y_train_list,axis=1)
		n,d,c=X_train.shape
		time_d =int(args.given_time*60/kdd_DATA.time_interval)
		X_train_2D = np.reshape(X_train,(n,time_d,-1,c))
		save_X_Y(args,X_train_2D, Y_train,sample_weights)
		return X_train_2D, Y_train,sample_weights
	else:
		return X,Y,sample_weights
	# return model, model_save_file
def generate_test_data(args):
	pass

def mape_and_kl_loss(y_true,y_pred):
	return losses.mape(y_true,y_pred)

def train_ensemble_deep_model_on_diff_sample(X,Y,model_save_file=None,model_create_Func=None,sample_weights=None,num_models=10):
	for i in range(num_models):
		X_tr,Y_tr,X_val,Y_val,s_weights=partition_train_val(X,Y,sample_weights,valid_on_high_weight=True)
		weight_file=model_save_file[:-3]+'_s'+str(i)+'.h5'
		print('ensemble train {} model {}'.format(i,weight_file))
		if 'LSTM'in model_save_file:
			X_train_channel_list=[]
			X_val_channel_list=[]
			for i in range(X_tr.shape[-1]):
				X_train_channel_list.append(X_tr[:,:,:,i])
				X_val_channel_list.append(X_val[:,:,:,i])
			train_deep_model(X_train_channel_list,Y_tr,X_val_channel_list,Y_val,weight_file,model_create_Func,s_weights)
		else:
			train_deep_model([X_tr],Y_tr, [X_val],Y_val,weight_file,model_create_Func,s_weights)


def predict_ensemble_deep_model_on_diff_sample(X,output_shape,model_save_file=None,model_create_Func=None,num_models=10):
	assert (type(X) is list)
	Y_p={}
	for i in range(num_models):
		# X_tr,Y_tr,X_val,Y_val,s_weights=partition_train_val(X,Y,sample_weights,valid_on_high_weight=True)
		weight_file=model_save_file[:-3]+'_s'+str(i)+'.h5'
		print('ensemble predict {} model {}'.format(i,weight_file))
		Y_p[i]=[]
		for j in range(len(X)):
			Y_p[i].append(model_predict([X[j]],output_shape,weight_file,model_create_Func=model_create_Func))
	Y_k=list(Y_p.keys())
	Y_k.sort()
	All=[Y_p[i] for i in Y_k]
	Y_arry=np.array(All)
	# ipdb.set_trace()

	Y=np.mean(Y_arry,axis=0)
	return Y

# def mape_and_variance_loss(Y_Pred,Y_True):


def train_deep_model(X_train,Y_train,X_val,Y_val,model_save_file=None,model_create_Func=None,sample_weights=None):
	# row =75
	# col =24
	assert(type(X_train) is list)
	x_ch_len = len(X_train)
	shape = X_train[0].shape
	if len(shape) ==3:
		n=shape[0]
		t=shape[1]
		l=shape[2]
		input_shape =(t,l)
	else:
		n,h,w,c =X_train[0].shape
		input_shape =(h, w,c)
	if model_save_file is not None:
		model_name=model_save_file
	else:
		model_name ='sample_conv.h5'
		# model_name ='sample_conv'
	n,out_n =Y_train.shape
	output_shape=(out_n,)
	# ip,out=kdd_deep_models.kdd_model(input_shape,output_shape)
	if model_create_Func is None:
		ip,out=kdd_deep_models.kdd_gated_model(input_shape,output_shape)
	else:
		ip,out=model_create_Func(input_shape,output_shape,x_ch_len)
	model=Model(ip,out)

	weight_h5_file='./weights/'+ model_name
	if os.path.isfile(weight_h5_file):
		try:
			model.load_weights(weight_h5_file)
		except:
			print ('the model {} can not  be loaded'.format(weight_h5_file))
			pass
	# from sklearn.model_selection import KFold
	# f = KFold(n_splits=2)
	# optimizer =RMSprop(lr = 1e-3)
	optimizer =Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.00001)
	model.compile(optimizer=optimizer, loss=mape_and_kl_loss, metrics=['MAPE'])#[metrics.mape])
	model.summary()
	best_model = ModelCheckpoint(weight_h5_file, verbose = 1, save_best_only = True)
	tensorboard= TensorBoard(log_dir='./logs',histogram_freq=0,write_graph=True,write_images=False)
	history=model.fit(x=X_train,y=Y_train, batch_size=96,epochs = 300, validation_data=(X_val,Y_val), \
					callbacks = [tensorboard,best_model],sample_weight=sample_weights,
					 shuffle=True)
	# history=model.fit(x=X_train,y=Y_train, epochs = 600, validation_split=0.05, \
	# 				callbacks = [tensorboard,best_model],sample_weight=np.array(sample_weights))
def train_GBM(X_train,Y_train,X_val,Y_val,model_save_file=None,sample_weights=None):
	from sklearn.ensemble import GradientBoostingRegressor  #GBM algorithm
	from sklearn import cross_validation, metrics   #Additional scklearn functions
	from sklearn.grid_search import GridSearchCV   #Perforing grid search
	gbms=[GradientBoostingRegressor(random_state=10) for i in range(Y_train.shape[1])]
	for i, gbm in enumerate(gbms):
		gbm.fit(X_train, Y_train[:,i], sample_weight=sample_weights)
	return gbms
def predict_GBM(X,gbms):
	assert type(gbms) is list
	y_Pl=[]
	# for i, gbm in enumerate(gbms):
	Y_ps=[gbm.predict(X) for gbm in gbms]
	return Y_ps

def model_predict(X_test,output_shape,model_weight_file=None,model_create_Func=None):
	global previous_model_file
	global previous_model
	assert(type(X_test) is list)
	x_ch_len = len(X_test)
	shape = X_test[0].shape
	if len(shape) ==3:
		n=shape[0]
		t=shape[1]
		l=shape[2]
		input_shape =(t,l)
	else:
		n,h,w,c =X_test[0].shape
		input_shape =(h, w,c)
	# ipdb.set_trace()
	# n,h,w,c =X_test.shape
	# # ip = Input(shape=(h, w,c))
	# input_shape =(h, w,c)
	if model_weight_file is not None:
		model_name=model_weight_file
	else:
		model_name ='sample_conv.h5'
	# model_name ='sample_conv'
	weight_h5_file='./weights/'+ model_name
	if previous_model_file is None or previous_model_file != model_name:
		print("model predicting ....")
		if model_create_Func is None:
			ip,out=kdd_deep_models.kdd_gated_model(input_shape,output_shape)
		else:
			ip,out=model_create_Func(input_shape,output_shape,x_ch_len)
		model=Model(ip,out)
		# ip,out=kdd_deep_models.kdd_gated_model(input_shape,output_shape)
		# model=Model(ip,out)
		previous_model_file =model_weight_file
		previous_model =model
		print("model predicting ...." +weight_h5_file)
		model.load_weights(weight_h5_file)
		# W=model.get_weight()
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
def partition_train_val(X,Y,weights=None,valid_on_high_weight=False):
	ratio =0.2
	n=X.shape[0]
	crossValid =True
	if crossValid:
		parts_num =int(1/ratio)
		part_idx =random.randint(0,parts_num)
		part_length =n/parts_num
		te_start_idx= part_idx *part_length
		te_end_idx  = (part_idx+1) *part_length
		part_idxs=range(te_start_idx,te_end_idx)
		X_te=X[part_idxs]
		Y_te=Y[part_idxs]
		X_tr=np.delete(X,part_idxs,axis=0)
		Y_tr=np.delete(Y,part_idxs,axis=0)
	else:
		parts_num =int(1/ratio)
		
		# part_idx =random.randint(0,n-parts_num)
		part_length =n/parts_num
		part_idx =n-part_length-1
		te_start_idx= part_idx
		te_end_idx  = part_idx+part_length
		# ipdb.set_trace()
		part_idxs=range(te_start_idx,te_end_idx)
		X_te=X[part_idxs]
		Y_te=Y[part_idxs]
		X_tr=np.delete(X,part_idxs,axis=0)
		Y_tr=np.delete(Y,part_idxs,axis=0)

	if valid_on_high_weight and weights is not None:
		max_w =np.max(weights)
		# ipdb.set_trace()
		te_w=weights[part_idxs]
		logic_idx =te_w==max_w
		# ipdb.set_trace()
		X_te=X_te[logic_idx]
		Y_te=Y_te[logic_idx]
		W_tr=np.delete(weights,part_idxs)
		return X_tr,Y_tr,X_te,Y_te,W_tr
	else:
		return X_tr,Y_tr,X_te,Y_te

if __name__ == "__main__":
	parser =argparse.ArgumentParser()
	parser.add_argument("-p","--phase", help="train or test",default='train')
	parser.add_argument("-r","--route_id", help="given a route_id for model to train",default='B-3')
	parser.add_argument("-i","--interval", help="feature inverval minutes",type=int,default=5)
	parser.add_argument("-c","--combine_features", nargs='*',dest='clist', help="list of combine features",default=argparse.SUPPRESS)
	parser.add_argument("-m","--model", help="select a deep model",default='default_model')
	parser.add_argument("-t","--given_time",help="preceeding data time length (in hour)",type=float,default =2)
	args=parser.parse_args()
	combined_feature_list =[]
	print (args)
	phase =args.phase
	if hasattr(args,'clist'):
		print (args.clist)
		combined_feature_list=args.clist
	# os._exit(0)
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

	##-------------------- concatenate routes information in Clist and train & predict in one model----- 
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
	# route_ids=['C-1','C-3','B-1','B-3','A-3']


	# route_ids=['A-2']
	if phase =='test':
		# route_id='A-3'
		test_combine_features={}
		test_combine_features['A-2']=[]
		test_combine_features['A-3']=[]
		test_combine_features['C-1']=[]
		test_combine_features['C-3']=[]
		test_combine_features['B-1']=[]
		test_combine_features['B-3']=[]
		model_create_Func,weight_file=get_model_and_weightFile(args)
		# route_id =args.route_id
		travel_time_test_info=kdd_DATA.read_test_data()
		predict_result ={}
		num_out_put =[6]
		for route_id in route_ids:
			test_combine_features[route_id].append(route_id)
			predict_result[route_id]={}
			for tt in test_time_list:
				start_time=datetime.strptime(tt[0],"%Y-%m-%d %H:%M:%S")
				end_time=datetime.strptime(tt[1],"%Y-%m-%d %H:%M:%S")
				travel_time_test_info=kdd_DATA.zerofill_missed_time_info(travel_times=travel_time_test_info,
																		 input_route_ids=[route_id],
																		 fill_route_ids=test_combine_features[route_id],
																		 start_time=start_time,end_time=end_time,
																		 phase='test')
				# ipdb.set_trace()
			args.route_id =route_id
			model_create_Func,weight_file=get_model_and_weightFile(args)
			# if not test_combine_features[route_id]:
			X_list ={}
			for i,r_id in enumerate(test_combine_features[route_id]):
				mat_test=kdd_DATA.get_test_feature_mat(travel_time_test_info,r_id,kdd_DATA.weather_test)
				time_list =list(mat_test.keys())
				time_list.sort()
				for i in range(len(time_list)):
					test_X = np.array(mat_test[time_list[i]])
					test_X=np.expand_dims(test_X,axis=2)
					test_X=np.expand_dims(test_X,axis=0)
					if time_list[i] not in X_list.keys():
						X_list[time_list[i]]=[]
					X_list[time_list[i]].append(test_X)
			# ipdb.set_trace()
			T_X=[]
			for i,time_i in enumerate(X_list.keys()):
				# test_X=np.concatenate(X_list[time_i],axis=2)
				T_X.append(np.concatenate(X_list[time_i],axis=2))
				# predict_result[route_id][time_i]=model_predict([test_X],num_out_put,weight_file,model_create_Func)
			A=predict_ensemble_deep_model_on_diff_sample(T_X,num_out_put,
														model_save_file=weight_file,
														model_create_Func=model_create_Func)
			# print(A.shape)
			# print(X_list.keys())
			# ipdb.set_trace()
			for i,time_i in enumerate(X_list.keys()):
				predict_result[route_id][time_i]=A[i]
			# predict_result[route_id][time_i]=predict_ensemble_deep_model_on_diff_sample(test_X,num_out_put,
			# 																		model_save_file=weight_file,
			# 																		model_create_Func=model_create_Func)
		write_submit_traj_file(predict_result)

	# import ipdb
	# ipdb.set_trace()
	if phase =='train':
		# route_id='A-2'
		model_create_Func,weight_file=get_model_and_weightFile(args)
		X_train_2D,Y_train,sample_weights=generate_train_data(args)
		# ipdb.set_trace()
		sample_weights=change_sample_weight(sample_weights,1,0.35)
		n,outp=Y_train.shape
		outshape =(outp,)
		X_tr,Y_tr,X_val,Y_val,W_tr=partition_train_val(X_train_2D,Y_train,weights=sample_weights,valid_on_high_weight=True)
		# X_train_2D,Y_train=shuffle(X_train_2D,Y_train,random_state=0)
		# sample_weights =W_tr
		from sklearn.utils import shuffle
		X_train_channel_list=[]
		X_val_channel_list=[]

		# if 'LSTM'in args.model:
		# 	# X_train_2D=X_train_2D[:,:,:,0]
		# 	for i in range(X_tr.shape[-1]):
		# 		X_train_channel_list.append(X_tr[:,:,:,i])
		# 		X_val_channel_list.append(X_val[:,:,:,i])
		# 	train_deep_model(X_train_channel_list,Y_tr,X_val_channel_list,Y_val,weight_file,model_create_Func,W_tr)
		# elif 'GBM' in args.model:
		# 	train_GBM(X_tr,Y_tr,X_val,Y_val,weight_file,W_tr)
		# else:
			# train_deep_model([X_tr],Y_tr, [X_val],Y_val,weight_file,model_create_Func,sample_weights )
			# X,Y,model_save_file=None,model_create_Func=None,sample_weights=None,num_models=10
		if 'GBM' in args.model:
			X_tr_1=np.reshape(X_tr,(X_tr.shape[0],-1))
			X_val_1=np.reshape(X_val,(X_val.shape[0],-1))
			gbms=train_GBM(X_tr_1,Y_tr,X_val_1,Y_val,weight_file,W_tr)
			Y_p=predict_GBM(X_val_1,gbms)
			Y_p=np.array(Y_p)
			Y_p=np.transpose(Y_p,(1,0))
		else: 
			train_ensemble_deep_model_on_diff_sample(X=X_train_2D,Y=Y_train,model_save_file=weight_file, \
													model_create_Func=model_create_Func,\
													sample_weights=sample_weights)
			Y_p=predict_ensemble_deep_model_on_diff_sample([X_val],outshape,weight_file,model_create_Func)
		# Y_p=model_predict([X_val],[6],weight_file,model_create_Func)
		print ('my mape = {}'.format(np.mean(np.abs((Y_val - Y_p) / Y_val)) * 100))





		# if 'LSTM'in args.model:
		# 	# X_train_2D=X_train_2D[:,:,:,0]
		# 	for i in range(X_train_2D.shape[-1]):
		# 		X_train_channel_list.append(X_train_2D[:,:,:,i])
		# 	train_deep_model(X_train_channel_list,Y_train,weight_file,model_create_Func,sample_weights )
		# else:
		# 	# pass
		# 	train_deep_model([X_train_2D],Y_train,weight_file,model_create_Func,sample_weights )
		#





		# train_deep_model(X_train_2D,Y_train,weight_file,model_create_Func)
		# y_p=model_predict(X_train_2D,outshape,saved_model_file_name)
		# train_deep_model(X_train_2D,Y_train,weight_file,model_create_Func)
		# Y_p=model_predict(X_train_2D,outshape,saved_model_file_name)
		# print("mape = {}".format(mean_absolute_error(Y_val,Y_p)))
		print("variance score = {}".format(explained_variance_score(Y_val,Y_p)))
		# print("Keras mape = {}".format(K.eval(metrics.mape(K.cast_to_floatx(Y_val),K.cast_to_floatx(Y_p)))))


		# from sklearn import linear_model 
		# clf= linear_model.MultiTaskLasso(alpha=0.1,max_iter=100)
		# clf.fit(X_train,Y_train)
		# Y_p =	clf.predict(X_train)
		# print("mean erros of route {}".format(route_id))
		# print(mean_absolute_error(Y_train,Y_p))
		# print(explained_variance_score(Y_train,Y_p))
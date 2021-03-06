import math
from datetime import datetime,timedelta
import numpy as np
# import autosklearn.regression
import sklearn.cross_validation
import sklearn.metrics
from sklearn import linear_model
from sklearn.cross_validation import cross_val_predict
import ipdb
import pandas as pd
import random
# import time
from datetime import datetime, date, time
import progressbar
import matplotlib.pyplot as plt
import math
file_suffix = '.csv'
path = '../../kdd_cup_data/dataSets/training/'  # set the data directory
test_path ='../../kdd_cup_data/dataSets/testing_phase1/'


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
nholidays=[('2016-09-15 00:00:00','2016-09-18 00:00:00'),
			('2016-10-1 00:00:00','2016-10-9 00:00:00')]
colum_num=128

def find_time_range(two_h_interval_list,start_time):
		for i,time_range in enumerate(two_h_interval_list):
			if start_time >=time_range[0] and start_time<time_range[1]:
				return i , time_range[0]

def one_hot(num,length):
	x = [0 for k in range(length)]
	x[num]=1
	return x
def time_in_range(start, end, x):
    """Return true if x is in the range [start, end]"""
    if start <= end:
    	# ipdb._set_trace()
        return start <= x < end
    else:
        return start <= x or x <= end

class time_recod_data():
	def __init__(self):
		self.link_tt={}
		self.link_st={}
		self.car_traj_time={}
		self.car_start_time={}
		self.weather={}
		
		# self.
	def zero_init(self,link_ids):
		for l_id in link_ids:
			self.link_tt[l_id] =[]
			self.link_st[l_id] =[]
			# self.link_tt(link_ids)class single_vehicle_time_record():
class record_based_traj_data():
	def __init__(self,traj_records,link_ids,weather_records):
		self.gather_traj_time_info(traj_records,link_ids,weather_records)
		self.holidays =[]
		self.X_num_colum=120
		self.weather_records = weather_records
		for s_time_str,e_time_str in nholidays:
			start_time= datetime.strptime(s_time_str, "%Y-%m-%d %H:%M:%S")
			end_time= datetime.strptime(e_time_str, "%Y-%m-%d %H:%M:%S")
			self.holidays.append((start_time,end_time))

	def time_in_holiday(self,given_time):
		in_range =False
		for start_time, end_time in self.holidays:
			# ipdb.set_trace()
			if time_in_range(start_time,end_time,given_time):
				in_range =True
				break
		return in_range

	def gather_traj_time_info(self,traj_records,link_ids,weather_records):
		self.traj_time_list=[time_traj_to_vector(each_record,link_ids,weather_records) for each_record in traj_records]

	# def generate_training_data_sample_on_route_id(self,route_id,time_slots_for_sample_weight=None,y_interval=20,p_hours=2):
	# 	colum_num =120
	# 	train_hours=p_hours
	# 	predict_hours=2
	# 	max_records=0
	# 	X=[]
	# 	Y=[]
	# 	if time_slots_for_sample_weight is not None:
	# 		s_weights =[]
	# 	else:
	# 		s_weights =None
	# 	print('generate data list...')
	# 	traj_routeID_time_list=[time_traj_obj for time_traj_obj in self.traj_time_list if time_traj_obj.route_id==route_id]
	# 	with progressbar.ProgressBar(max_value=len(self.traj_time_list)) as bar:
	
	def set_train_generator(self,route_id, predict_all_routes=False,
							time_slots_for_sample_weights=None, 
							remove_holiday_data=True,
							y_interval=20,p_hours=2,data_dim=4):
		self.route_id=route_id
		self.time_slots_for_sample_weights=time_slots_for_sample_weights
		self.remove_holiday_data=remove_holiday_data
		self.y_interval=y_interval
		self.p_hours=p_hours
		self.data_dim =data_dim
		self.predict_all_routes =predict_all_routes
		self.X_num_colum =120 if not predict_all_routes else 480
		# ipdb.set_trace()
		if self.remove_holiday_data:
			if  not predict_all_routes:
				self.traj_routeID_list=[time_traj_obj for time_traj_obj in self.traj_time_list 
										if time_traj_obj.route_id==self.route_id \
										and not self.time_in_holiday(time_traj_obj.trace_start_time)]
			else:
				self.traj_routeID_list=[time_traj_obj for time_traj_obj in self.traj_time_list 
										if not self.time_in_holiday(time_traj_obj.trace_start_time)]

		else:
			if  not predict_all_routes:
				self.traj_routeID_list=[time_traj_obj for time_traj_obj in self.traj_time_list if time_traj_obj.route_id==self.route_id]
			else:
				self.traj_routeID_list=[time_traj_obj for time_traj_obj in self.traj_time_list]
									# if time_traj_obj.route_id==self.route_id or time_traj_obj.route_id=='C-1']#and time_traj_obj.route_id=='C-1']
		# ipdb.set_trace()
	def train_val_partition(self,part_ratio =0.2,rand_seed=None):
		num_d_points 	= 	len(self.traj_routeID_list)
		parts_num 		=	int(1/part_ratio)
		part_length 	=	num_d_points/parts_num
		part_idx 		=	num_d_points-part_length-1
		self.rand_seed  =   rand_seed
		if rand_seed is not None:
			random.seed(rand_seed)
		# ipdb.set_trace()
		val_start_idx	= 	random.randint(0,part_idx) 
		val_end_idx  	= 	val_start_idx+part_length
		# ipdb.set_trace()
		part_idxs		=	range(val_start_idx,val_end_idx)
				
		self.train_list =   [traj_obj for i,traj_obj in enumerate(self.traj_routeID_list) if i not in part_idxs]
		self.val_list 	=   [traj_obj for i,traj_obj in enumerate(self.traj_routeID_list) if i in part_idxs]
		# ipdb.set_trace()
		# return self.train_list, self.val_list


	def get_train_data_shape(self):
		h=self.X_num_colum
		w=len(self.traj_time_list[0].vector[0])+1
		c=1
		if self.data_dim==4:
			data_shape=(h,w,c)
		else:
			data_shape=(h,w)
		# ipdb.set_trace()
		return data_shape

	def generator(self,traj_obj_list,yield_weight =True,rand_seed=None,phase='train'):
	
		route_ids=[self.route_id] if not self.predict_all_routes else ['A-2','A-3','B-1','B-3','C-1','C-3']
		colum_num =self.X_num_colum
		train_hours=self.p_hours
		predict_hours=2
		max_records=0
		batch_size =96
		range_preceding_minutes=30 
		# ipdb.set_trace()
		
		# print('generate data list...')
		traj_routeID_time_list=traj_obj_list
		len_sample =len(traj_routeID_time_list)
		if rand_seed is not None:
			random.seed(rand_seed)
		else:
			random.seed(self.rand_seed)
		while True:
			X=[]
			Y=[]

			if self.time_slots_for_sample_weights is not None:
				s_weights =[]
			else:
				s_weights =None

			loop =0
			while len(X)<batch_size:
				# print ('loop ={} , len(X) = {}'.format(loop,len(X)))
				loop+=1
				each_X =[]
				each_Y =[]
				idx =random.randint(0,len_sample-1)
				preceeding_minutes =random.randint(0,range_preceding_minutes-1)
				rand_seconds  =random.randint(0,59)
				s_time =traj_routeID_time_list[idx].trace_start_time
				x_start_time= datetime(s_time.year, s_time.month, s_time.day,
													s_time.hour, s_time.minute, s_time.second)
				# if phase =='train':
				x_start_time -=timedelta(minutes=preceeding_minutes,seconds=rand_seconds)
				# x_start_time -=timedelta(seconds=rand_seconds)
				x_end_time=x_start_time+timedelta(hours=train_hours)
				# ipdb.set_trace()

				#--------------------------- prepare X -----------------------------------
				n_pos =1
				r_id_count =0
				count_record =0

				# backward search
				# if phase=='validation':
				while idx-n_pos>=0:
						cur_time =traj_routeID_time_list[idx-n_pos].trace_start_time
						cur_record_obj=traj_routeID_time_list[idx-n_pos]
						# if traj_routeID_time_list[idx-n_pos].route_id ==self.route_id and \
						# to_end_time_sec_diff=(x_end_time-cur_time).seconds
						# to_end_time_sec_diff=(cur_time-x_start_time).seconds
						to_end_time_sec_diff=int((cur_time-x_start_time).seconds/60)
						if cur_time >=x_start_time and \
											cur_time <x_end_time:
							
							if not(self.remove_holiday_data and self.time_in_holiday(cur_time)):
								# cur_record_obj.re_generate_vector() # add some randomness to travel_time
								each_X.append(cur_record_obj.vector[0]+[to_end_time_sec_diff])
								r_id_count+=1
								
						if cur_time <x_start_time \
								   or r_id_count>=colum_num:
										break;
						n_pos+=1
				each_X.reverse()	

				# forward search
				pos=0
				while idx+pos<len(traj_routeID_time_list):
					cur_time =traj_routeID_time_list[idx+pos].trace_start_time
					to_end_time_sec_diff=(cur_time-x_start_time).seconds
					cur_record_obj=traj_routeID_time_list[idx+pos]
					# if traj_routeID_time_list[idx+pos].route_id ==self.route_id and \
					if cur_time >=x_start_time and \
										cur_time <x_end_time:
						
						if not(self.remove_holiday_data and self.time_in_holiday(cur_time)):
								# cur_record_obj.re_generate_vector() # add some randomness to travel_time
								each_X.append(cur_record_obj.vector[0]+[to_end_time_sec_diff])
								r_id_count+=1
							
					# if cur_time >=x_end_time \
					# 		   and cur_record_obj.route_id ==self.route_id \
					# 		   or r_id_count>=colum_num:
					# 				break;
					if cur_time >=x_end_time \
							   or r_id_count>=colum_num:
									break;
					pos+=1
					count_record+=1

				max_records =r_id_count if max_records <r_id_count else max_records
				# print ('max_records ={}'.format(max_records))
				# -----------------------prepare Y--------------------------------------
				num_y_interval =predict_hours*60/self.y_interval
				for route_id in route_ids:
					y_start_time =x_end_time
					# y_start_time =x_start_time
					y_j=0
					for j in range(num_y_interval):
						y_end_time =y_start_time+timedelta(minutes=self.y_interval)
						y_temp=[]
						# y_idx=idx+pos+y_j
						# y_idx=idx+pos+y_j
						# print('indx ={} and len_sample= {}'.format(idx+pos+y_j,len_sample))
						while idx+pos+y_j<len_sample:
						# while idx-n_pos+1+y_j<len_sample:
							y_record_obj=traj_routeID_time_list[idx+pos+y_j]
							# y_record_obj=traj_routeID_time_list[idx-n_pos+1+y_j]
							# ipdb.set_trace()
							if y_record_obj.route_id==route_id:

								if y_record_obj.trace_start_time>y_end_time:
									break
								if y_record_obj.trace_start_time>=y_start_time:
									y_temp.append(y_record_obj.travel_time)
							y_j+=1
						y_start_time=y_end_time
						if len(y_temp) ==0:
							each_Y.append(0)
						else:
							each_Y.append(np.mean(y_temp))
					# Sipdb.set_trace()

				#--------------------------- prepare sample weights ----------------------
				if len(each_X) >0: 
					if self.time_slots_for_sample_weights is not None:
						num_slots =len(self.time_slots_for_sample_weights)
						s_weight =0.95
						time_in_mid_y =y_end_time-timedelta(minutes=predict_hours*60.0/2.0)
						for t1,t2 in self.time_slots_for_sample_weights:
							if phase=='train':
								st_time=time(t1.hour-1,t1.minute+55,t1.second)
								e_time=time(t2.hour,t2.minute+5,t2.second)
							else:
								st_time=time(t1.hour,t1.minute+50,t1.second)
								e_time=time(t2.hour-1,t2.minute+10,t2.second)

							if time_in_mid_y.time()>=st_time \
								and time_in_mid_y.time()<e_time:
									s_weight = 1
									break
						s_weights.append(s_weight)
					if phase =='validation':
						if s_weight==1:
							X.append(each_X)
							Y.append(each_Y)
					else:
						X.append(each_X)
						Y.append(each_Y)

			# print('yield one batch...')
			d_len=len(X)
			v_len=len(X[0][0])
			# ipdb.set_trace()
			X_array=np.zeros((d_len,colum_num,v_len))
			for i in range(d_len):
				for j in range(len(X[i])):
					X_array[i,colum_num-len(X[i])+j]=np.array(X[i][j])
			X_train=np.expand_dims(X_array,axis=3) if self.data_dim ==4 else X_array
			# print('g data_dim={}'.format(X_train.shape))
			Y=np.array(Y)
			if yield_weight:
				s_weights=np.array(s_weights)
				yield (X_train,Y, s_weights)
			else:
				yield (X_train,Y)



	def generate_training_data(self,route_id,time_slots_for_sample_weights=None, remove_holiday_data=True,y_interval=5,p_hours=2):
		colum_num =128
		train_hours=p_hours
		predict_hours=2
		max_records=0
		X=[]
		Y=[]
		if time_slots_for_sample_weights is not None:
			s_weights =[]
		else:
			s_weights =None
		print('generate data list...')
		traj_routeID_time_list=[time_traj_obj for time_traj_obj in self.traj_time_list if time_traj_obj.route_id==route_id]
		self.traj_time_list=traj_routeID_time_list
		with progressbar.ProgressBar(max_value=len(self.traj_time_list)) as bar:
			for i,time_traj_vector_obj in enumerate(self.traj_time_list):
				
				preceeding_minutes=range(15)
				for p_m in preceeding_minutes:
					each_X =[]
					each_Y =[]
					s_time =time_traj_vector_obj.trace_start_time
					minutes = s_time.minute
					# ipdb.set_trace()
					x_start_time= datetime(s_time.year, s_time.month, s_time.day,
												s_time.hour, s_time.minute, 0)
					x_start_time -=timedelta(minutes=p_m)
					x_end_time=x_start_time+timedelta(hours=train_hours)
					pos =0
					r_id_count =0
					count_record =0
					while i+pos<len(self.traj_time_list):
						cur_time =self.traj_time_list[i+pos].trace_start_time
						# ipdb.set_trace()
						if self.traj_time_list[i+pos].route_id ==route_id and \
							cur_time >=x_start_time and \
							cur_time <x_end_time:
							if remove_holiday_data:
								# ipdb.set_trace()
								if not self.time_in_holiday(cur_time):
									each_X.append(self.traj_time_list[i+pos].vector)
									r_id_count+=1
							else:
								each_X.append(self.traj_time_list[i+pos].vector)
								r_id_count+=1
						
						if cur_time >x_end_time \
						    and self.traj_time_list[i+pos].route_id ==route_id    \
							or r_id_count>=colum_num:
								break;
						pos+=1
						count_record+=1
						
					# max_records =count_record if max_records <count_record else max_records
					max_records =r_id_count if max_records <r_id_count else max_records
					num_y_interval =predict_hours*60/y_interval
					y_start_time =x_end_time
					
					y_j=0
					for j in range(num_y_interval):
						y_end_time =y_start_time+timedelta(minutes=y_interval)
						y_temp=[]
						while i+pos+y_j<len(self.traj_time_list):
							if self.traj_time_list[i+pos+y_j].route_id ==route_id:
								if self.traj_time_list[i+pos+y_j].trace_start_time>y_end_time:
									break
								if self.traj_time_list[i+pos+y_j].trace_start_time>=y_start_time:
									y_temp.append(self.traj_time_list[i+pos+y_j].travel_time)
							y_j+=1
						y_start_time=y_end_time
						if len(y_temp) ==0:
							each_Y.append(0)
						else:
							each_Y.append(np.mean(y_temp))
					
					if len(each_X) >0: 
						X.append(each_X)
						Y.append(each_Y)
						if time_slots_for_sample_weights is not None:
							num_slots =len(time_slots_for_sample_weights)
							s_weight =0.7
							time_in_mid_y =y_end_time-timedelta(minutes=predict_hours*60.0/2.0)
							for t1,t2 in time_slots_for_sample_weights:
								st_time=time(t1.hour,t1.minute+15,t1.second)
								e_time=time(t2.hour-1,t2.minute+45,t2.second)
								if time_in_mid_y.time()>=st_time and \
											time_in_mid_y.time()<e_time:
									s_weight = 1
									break
							s_weights.append(s_weight)


				bar.update(i)
				# print ('Add {} of {}'.format(i,len(self.traj_time_list)))
		# print ('rout_id = {} max record ={}'.format(route_id,max_records))
		# ipdb.set_trace()
		d_len=len(X)
		# ipdb.set_trace()
		v_len=len(X[0][0][0])
		# ipdb.set_trace()
		X_array=np.zeros((d_len,colum_num,v_len))
		# print('converting data list to numpy array...')
		with progressbar.ProgressBar(max_value=d_len) as bar:
			for i in range(d_len):
				for j in range(len(X[i])):
					# ipdb.set_trace()
					X_array[i,j]=np.array(X[i][j][0])
				bar.update(i)
			# 	# ipdb.set_trace()
		# ipdb.set_trace()
		Y=np.array(Y)
		s_weights=np.array(s_weights)
		return X_array,Y, s_weights
		# s_weight=1 if last.time()>=t_start_1.time() and last.time()<t_end_1.time() or \
		# 				last.time()>=t_start_2.time() and last.time()<t_end_2.time() \
		# 				else 0.1



		# print (len(traj_data)) 
		# for i in range(len(traj_data)):
	def generate_test_data(self,route_id,test_time_list=test_time_list):
		self.X_num_colum =120 if route_id is not None else 480
		colum_num =self.X_num_colum
		# ipdb.set_trace()
		test_data_list ={}
		test_data_X={}
		route_records  ={}
		# ipdb.set_trace()
		two_h_interval=[]
		v_len =len(self.traj_time_list[0].vector[0])
		for time_str in test_time_list:
			t_start =datetime.strptime(time_str[0], "%Y-%m-%d %H:%M:%S")
			t_end   =datetime.strptime(time_str[1], "%Y-%m-%d %H:%M:%S")
			two_h_interval.append((t_start,t_end))
		# route_records =[each_record for each_recod in self.traj_time_list if each_record.route_id ==route_id]
		for i, each_record in enumerate(self.traj_time_list):
			if route_id is not None:
				if each_record.route_id ==route_id:
					route_records[each_record.trace_start_time]=each_record
			else:
				route_records[each_record.trace_start_time]=each_record
		start_times =list(route_records.keys())
		start_times.sort()
		for start_time in start_times:
			# ipdb.set_trace()
			i,time_range=find_time_range(two_h_interval,start_time)
			# ipdb.set_trace()
			# time_diff=(time_range+timedelta(hours=2)-start_time).seconds
			time_diff=(start_time-time_range).seconds
			# ipdb.set_trace()
			if time_range not in test_data_list.keys():
				test_data_list[time_range]=[]
			route_records[start_time].vector[0]+=[time_diff]
			test_data_list[time_range].append(route_records[start_time])


		for timeSlot_key in list(test_data_list.keys()):
			X=np.zeros((colum_num,v_len+1))
			d_len=len(test_data_list[timeSlot_key])
			for i in range(d_len):
				X[colum_num-d_len+i,:]=np.array(test_data_list[timeSlot_key][i].vector[0])
				# ipdb.set_trace()
			test_data_X[timeSlot_key]=X
		# ipdb.set_trace()
		return test_data_X
class links_to_image_convertor():
	def __init__(self,link_seq_ids,route_id,time_col,start_time,end_time):
		self.route_id =route_id
		self.start_time =start_time
		self.end_time   =end_time
		self.num_ch =2
		self.time_col=time_col
		self.ink_seq_ids=link_seq_ids
		len_link=len(link_seq_ids)
		self.time_interval_second=(end_time-start_time).seconds/float(time_col)
		self.channels =[]
		for c in range(self.num_ch):
			link_raw_container ={}
			for link in link_seq_ids:
				link_raw_container[link]=[[] for c in range(time_col)]
			self.channels.append(link_raw_container)
		# ipdb.set_trace()
	def compute_col_idx(self,link_enter_time):
		# if self.start_time<=link_enter_time <=self.end_time:
		idx=math.floor((link_enter_time-self.start_time).seconds/float(self.time_interval_second))
		idx =idx if self.start_time<=link_enter_time <=self.end_time else -1
		return  idx
	def fill_record(self,time_traj_to_vector_obj):
		if time_traj_to_vector_obj.route_id !=self.route_id:
			return
		for link in time_traj_to_vector_obj.used_link_ids:
			l_st=time_traj_to_vector_obj.link_start_t[link]
			l_tt=time_traj_to_vector_obj.link_travel_t[link]
			col_idx=int(self.compute_col_idx(l_st))
			if col_idx ==-1:
				continue
			else:
				self.channels[0][link][col_idx].append(l_tt)
				self.channels[1][link][col_idx].append(l_st)
			# except:
				# ipdb.set_trace()
	def to_image(self):
		row=len(self.channels[0])
		col=self.time_col
		im_list =[]
		for i in range(3):
			im_list.append(np.zeros((row,col)))
		for i,link in enumerate(self.ink_seq_ids):
			row_mean_tt_list	=[0 if len(self.channels[0][link][col_idx])==0 else np.mean(self.channels[0][link][col_idx]) for col_idx in range(self.time_col)]
			row_count_tt_list	=[len(self.channels[0][link][col_idx]) for col_idx in range(self.time_col)]
			row_std_tt_list		=[0 if len(self.channels[0][link][col_idx])==0 else np.std(self.channels[0][link][col_idx]) for col_idx in range(self.time_col)]
			im_list[0][i]		=np.array(row_mean_tt_list)
			im_list[1][i]		=np.array(row_count_tt_list)
			im_list[2][i]		=np.array(row_std_tt_list)
		return im_list
class test_links_to_image_convertor():
	def __init__(self):
		r_id ='B-1'
		link_seq={}
		link_seq['C-1']=['115','102','109','104','112','111','103','116','101','121','106','113']
		link_seq['B-1']=['105','100','111','103','116','101','121','106','113']

		start_time=datetime(2016,10,5,9,0,0)
		end_time=start_time+timedelta(hours=2)
		l_im_obj=links_to_image_convertor(link_seq[r_id],r_id,30,start_time,end_time)
		kd_obj=kdd_data()
		traj_records,link_ids,weath_records=kd_obj.get_traj_raw_records_train(r_id)
		RRBTD=record_based_traj_data(traj_records,link_ids,weath_records)
		for rd in RRBTD.traj_time_list:
			if rd.trace_start_time >=start_time and rd.trace_start_time <=end_time:
				l_im_obj.fill_record(rd)
		im_list=l_im_obj.to_image()
		fig, ax = plt.subplots(3)
		ax[0].imshow(im_list[0])
		ax[1].imshow(im_list[1])
		ax[2].imshow(im_list[2])
		# ipdb.set_trace()
		plt.show()









class temperal_spatial_traj_data(record_based_traj_data):
	def __init__(self,traj_records,link_ids,weather_records):
		super(temperal_spatial_traj_data,self).__init__(traj_records,link_ids,weather_records)

	def generator(self,traj_obj_list,yield_weight =True,rand_seed=None,phase='train'):
		pass



			

# trace_start_time = each_traj[3]
			# trace_start_time = datetime.strptime(trace_start_time, "%Y-%m-%d %H:%M:%S")


class time_traj_to_vector():
	def __init__(self,traj_data,link_ids,weather_records):
		traj_info = traj_data.replace('"', '').split(',')
		self.link_ids =link_ids
		self.weather_records 	= weather_records
		self.intersection_id 	= traj_info[0]
		self.tollgate_id 		= traj_info[1]
		self.route_id 			= self.intersection_id + '-' + self.tollgate_id
		# ipdb.set_trace()
		self.trace_start_time 	= datetime.strptime(traj_info[3], "%Y-%m-%d %H:%M:%S")
		self.travel_time  		= float(traj_info[-1]) # travel time
		self.link_travel_t={}
		self.link_start_t={}
		self.link_time_diff={}
		self.vector =None
		for l_id in self.link_ids:
			self.link_travel_t[l_id]=0
			self.link_start_t[l_id]=datetime(2016,1,1,0,0,0)
			self.link_time_diff[l_id]=0
		link_seq= traj_info[4].split(';')
		self.used_link_st_time=[]
		self.used_link_ids=[]
		for link in link_seq:
			info = link.split('#')
			self.link_travel_t[info[0]]=float(info[2])
			self.link_start_t[info[0]]=datetime.strptime(info[1], "%Y-%m-%d %H:%M:%S")
			self.used_link_st_time.append(self.link_start_t[info[0]])
			self.used_link_ids.append(info[0])
		self.used_link_st_time.sort()
		# ipdb.set_trace()
		self.vector=self.to_vector()
	def re_generate_vector(self):
		self.vector=self.to_vector(add_rand_to_travel_time=True)
	def link_to_vector(self):
		min_link_start_time=self.used_link_st_time[0]
		ids =self.link_travel_t.keys()
		ids.sort()
		link_tt_v=[]
		link_st_v=[]
		link_time_diff=[]
		for l_id in ids:
			link_tt_v.append((self.link_travel_t[l_id]+1))
			if self.link_start_t[l_id] is not None:
				hour =(self.link_start_t[l_id].hour+1)
				minute =(self.link_start_t[l_id].minute+1)
				second =(self.link_start_t[l_id].second+1)
			else:
				hour=0
				minute=0
				second=0
			link_st_v+=[hour,minute,second]
			if l_id in self.used_link_ids:
				t_diff=self.link_start_t[l_id] - min_link_start_time
				link_time_diff+=[t_diff.seconds]
			else:
				link_time_diff+=[0]

		vector=link_tt_v+link_st_v+link_time_diff
		return vector
	def get_weather_data(self,current_time):
		w_times=list(self.weather_records.keys())
		w_times.sort()
		for i,w_time in enumerate(w_times):
			if i<len(w_times)-1:
				if current_time >= w_time and current_time <w_times[i+1]:
					weather_feature = self.weather_records[w_time] +self.weather_records[w_times[i+1]]
					return weather_feature
		weather_feature = self.weather_records[w_time] +self.weather_records[w_time]
		return weather_feature
	def to_vector(self,add_rand_to_travel_time=False):
		if self.vector is not None:
			return self.vector
		# ipdb.set_trace()
		# self.vector=[one_hot(self.intersecID_to_int(self.intersection_id),4)+one_hot(int(self.tollgate_id),4)
		# 			+one_hot(self.trace_start_time.weekday()+1,9)+one_hot(self.trace_start_time.hour+1,25)
		# 			+one_hot(self.trace_start_time.minute+1,61)+[self.trace_start_time.second]+[self.travel_time]+self.link_to_vector()]
		# rand_time=random.randint(0,5)
		# rand_time =rand_time if random.randint(0,1)==0 else -1*rand_time
		# rand_time = rand_time if add_rand_to_travel_time else 0
		rand_time=0
		self.vector=[[self.travel_time+rand_time]+one_hot(self.intersecID_to_int(self.intersection_id),4)+one_hot(int(self.tollgate_id),4)
					+one_hot(self.trace_start_time.weekday()+1,9)+[(self.trace_start_time.hour+1)]
					+[(self.trace_start_time.minute+1)]+[(self.trace_start_time.second+1)]+self.link_to_vector()
					+self.get_weather_data(self.trace_start_time)]
		# ipdb.set_trace()
		return self.vector
	def intersecID_to_int(self,intersec_id):
		if intersec_id=='A':
			return 0
		elif intersec_id=='B':
			return 1
		elif intersec_id =='C':
			return 2
		else:
			raise InputError()

class kdd_data():
	def __init__(self,interval =5):
		self.time_interval = interval # 10 minutes interaval
		
		# self.travel_times =self.zerofill_missed_time_info(self.travel_times)
	def read_train_data(self):
		time_file = path+'trajectories(table 5)_training'+file_suffix
		vol_file =path+'volume(table 6)_training'+file_suffix
		road_link_file=path+'links (table 3)'+file_suffix 
		weather_file =path +'weather (table 7)_training_update'+file_suffix
		traj_data,vol_data,link_ids_data,weather_records=self.load_data(time_file,vol_file,road_link_file,weather_file)
		self.link_ids=self.parse_road_link_ids(link_ids_data)
		self.weather_train =self.parse_weather_info(weather_records)
		self.travel_times=self.format_data_in_timeInterval(traj_data,vol_data)
		

	def read_test_data(self):
		test_time_file = test_path+'trajectories(table 5)_test1'+file_suffix
		test_vol_file = test_path+'volume(table 6)_test1'+file_suffix
		road_link_file=path+'links (table 3)'+file_suffix
		weather_file =test_path +'weather (table 7)_test1'+file_suffix
		traj_data,vol_data,link_ids_data,weather_records=self.load_data(test_time_file,test_vol_file,road_link_file,weather_file)
		self.link_ids=self.parse_road_link_ids(link_ids_data)
		self.weather_test=self.parse_weather_info(weather_records)
		return self.format_data_in_timeInterval(traj_data,vol_data)
	
	
	def get_test_feature_mat(self,travel_time_info,route_id,weather_data):
		

		# datetime.strptime(trace_start_time, "%Y-%m-%d %H:%M:%S")
		two_h_interval =[]
		dstart =[]
		dend=[]
		for time_str in test_time_list:
			dstart =datetime.strptime(time_str[0], "%Y-%m-%d %H:%M:%S")
			dend   =datetime.strptime(time_str[1], "%Y-%m-%d %H:%M:%S")
			two_h_interval.append((dstart,dend))
		D=travel_time_info[route_id]
		time_windows =list(D.keys())
		time_windows.sort()
		mat = {}
		# import ipdb
		# ipdb.set_trace()
		for t_w in time_windows:
			t_record = D[t_w]
			vect=self.convert_windowInfo_to_vector(t_record,t_w,weather_data)
			start_times =list(t_record.car_start_time.values())
			start_times.sort()
			if len(start_times)==0:
				start_times.append(t_w)
			if two_h_interval is None:
				import ipdb
				ipdb.set_trace()
			# print (two_h_interval)

			try:
				i,time_range=find_time_range(two_h_interval,start_times[0])
			except:
				import ipdb
				ipdb.set_trace()

			if time_range not in mat.keys():
				mat[time_range]=[]
			mat[time_range].append(vect)
			# print vect
		return mat
	def select_timewindos_data_and_zerofill(self,travel_times,route_id =None,start_time=None,end_time =None):

		current_time = start_time

		# 2 . get all routes link ids
		link_tt_ids={}
		for rd_ids in travel_times.keys():
			first_time =  next(iter(travel_times[rd_ids]))
			link_tt_ids[rd_ids]=list(travel_times[rd_ids][first_time].link_tt.keys())
		added_time =timedelta(minutes=self.time_interval)
		if route_id is None:
			r_ids =list(travel_times.keys())
		else:
			r_ids=[route_id]
			print (r_ids)		
		travel_times_new={}
		for r_id in r_ids:
			travel_times_new[r_id] = {}
			time_windows  =list(travel_times[r_id].keys())
			#print(time_windows)
			for temp_time_window in time_windows:
				if temp_time_window.time() < end_time.time() and temp_time_window.time() >= start_time.time():
					temp_object = travel_times[r_id][temp_time_window]
					travel_times_new[r_id][temp_time_window] = temp_object
			#print(test)
			print(travel_times_new[r_id])
		travel_times = travel_times_new

		print('start to fill the missed time interval ... data')
		while current_time < end_time:
			# for r_id in travel_times.keys():
			for r_id in r_ids:
				# print (r_id)
				time_windows  =list(travel_times[r_id].keys())   # this returns the time windows in the test data for each combination 'r_id'
				#print(time_windows)

				if current_time not in time_windows:
					t_record =time_recod_data()
					#print(current_time)
					t_record.zero_init(link_tt_ids[rd_ids])
					travel_times[r_id][current_time] =t_record
					#print (current_time)
			current_time+=added_time
		return travel_times	
	def zerofill_missed_time_info(self,travel_times,input_route_ids=None,fill_route_ids =None,start_time=None,end_time =None,phase='train'):
		# 1. find min start-time and max-end time across all roud_ids
		# import ipdb
		# ipdb.set_trace()
		if start_time==None or end_time ==None:
			start_time = datetime(2100,1,1,1,0)
			end_time  =datetime(1900,1,1,1,0)
			if input_route_ids is not None:
				r_list =input_route_ids
			else:
				r_list =travel_times.keys()
			for r_id in r_list:
				time_windows  =list(travel_times[r_id].keys())
				time_windows.sort()
				start_time = time_windows[0] if start_time > time_windows[0] else start_time
				end_time = time_windows[-1] if end_time < time_windows[-1] else end_time

			# if route_ids is not None:
			# 	time_windows  =list(travel_times[route_id].keys())
			# 	time_windows.sort()
			# 	start_time = time_windows[0] if start_time > time_windows[0] else start_time
			# 	end_time = time_windows[-1] if end_time < time_windows[-1] else end_time
			# else:
			# 	for r_id in travel_times.keys():
			# 		time_windows  =list(travel_times[r_id].keys())
			# 		time_windows.sort()
			# 		start_time = time_windows[0] if start_time > time_windows[0] else start_time
			# 		end_time = time_windows[-1] if end_time < time_windows[-1] else end_time
		current_time = start_time


		# 3 . get all routes link ids
		link_tt_ids={}
		for rd_ids in travel_times.keys():
			first_time =  next(iter(travel_times[rd_ids]))
			link_tt_ids[rd_ids]=list(travel_times[rd_ids][first_time].link_tt.keys())
		added_time =timedelta(minutes=self.time_interval)
		if fill_route_ids is None:
			r_ids =list(travel_times.keys())
		else:
			r_ids=fill_route_ids
			print (r_ids)

		# Remove the time windows that are not in given range, so that all routes' time windows will be the same for
		# next step of composting them together as different channel data. 
		if phase =='train':
			for r_id in fill_route_ids:
				time_windows  =travel_times[r_id].keys()
				for time_w in time_windows:
					if time_w <start_time or time_w>= end_time:
						del travel_times[r_id][time_w]


		print('start to fill the missed time interval ... data')
		while current_time < end_time:
			# for r_id in travel_times.keys():
			for r_id in r_ids:
				# print (r_id)
				time_windows  =list(travel_times[r_id].keys())
				if current_time not in time_windows:
					t_record =time_recod_data()
					t_record.zero_init(link_tt_ids[rd_ids])
					travel_times[r_id][current_time] =t_record
					# print (current_time)
			current_time+=added_time
		for r_id in r_ids:
			print( r_id + ' lentgh = {}'.format(len(list(travel_times[r_id].keys()))))
		# import ipdb
		# ipdb.set_trace()

		return travel_times

	def parse_road_link_ids(self,link_ids_data):
		link_ids =[]
		for i in range(len(link_ids_data)):
			each_line = link_ids_data[i].replace('"', '').split(',')
			link_ids.append(each_line[0])
		# print (len(link_ids))
		return link_ids

	def parse_weather_info(self,weather_data):
		weather_dict={}
		for i in range(len(weather_data)):
			each_3h_weather =weather_data[i].replace('"', '').split(',')
			t=each_3h_weather[0]
			year_m_d = datetime.strptime(t, "%Y-%m-%d")
			hour =each_3h_weather[1]
			# print (year_m_d)
			# print (hour)
			time_key =datetime(int(year_m_d.year),int(year_m_d.month),int(year_m_d.day),int(hour),0,0)
			data =[float(each_3h_weather[i]) for i in range(2,len(each_3h_weather))]
			weather_dict[time_key]=data
			# ipdb.set_trace()
		return weather_dict
	def get_weather_data(self,weather_data,current_time):
		w_times=list(weather_data.keys())
		w_times.sort()
		for i,w_time in enumerate(w_times):
			if i<len(w_times)-1:
				if current_time >= w_time and current_time <w_times[i+1]:
					weather_feature = weather_data[w_time] +weather_data[w_times[i+1]]
					return weather_feature
		weather_feature = weather_data[w_time] +weather_data[w_time]
		return weather_feature


		
	def load_data(self,time_file,vol_file,road_link_file,weather_file):
		# Step 1: Load trajectories
		fr = open(time_file, 'r')
		fr.readline()  # skip the header
		traj_data = fr.readlines()
		fr.close()
		#step 2: load valume
		fr = open(vol_file, 'r')
		fr.readline()  # skip the header
		vol_data = fr.readlines()
		fr.close()

		fr =open(road_link_file,'r')
		fr.readline() # skip the header
		link_ids=fr.readlines()
		fr.close()

		fr =open(weather_file,'r')
		fr.readline() # skip the header
		weather_data=fr.readlines()
		fr.close()
		return traj_data, vol_data, link_ids, weather_data

	def get_feature_matrix(self,travel_times_struct,route_id,weather_data):
		D=travel_times_struct[route_id]
		# import ipdb
		# ipdb.set_trace()
		time_windows =list(D.keys())
		time_windows.sort()
		mat=[]
		for t_w in time_windows:
			vect=self.convert_windowInfo_to_vector(D[t_w],t_w,weather_data)
			mat.append(vect)
			# print vect
		return mat,time_windows
	def interpolate_zerolist(self, list_Y):
		# all_Y  =list(list_Y)
		all_Y=list_Y#[0 for i in range(len(list_Y))]
		non_zero_y_idx =[]
		len_f=len(list_Y)
		non_zero_y_idx =[i for i, e in enumerate(list_Y) if e!=0]
		# print len(non_zero_y_idx)
		first_nonzero_idx =non_zero_y_idx[0]
		left_idx  =0
		right_idx =1
		for c_idx in range(len_f):
			if list_Y[c_idx]==0:
				if c_idx<first_nonzero_idx:
					all_Y[c_idx]=list_Y[first_nonzero_idx]
				else:
					while not (c_idx> non_zero_y_idx[left_idx] and c_idx <non_zero_y_idx[right_idx]) and right_idx<len(non_zero_y_idx)-1:
						left_idx+=1
						right_idx+=1
					l_idx=non_zero_y_idx[left_idx]
					r_idx=non_zero_y_idx[right_idx]
					# if c_idx >non_zero_y_idx[right_idx]:
					w1=0 if c_idx >non_zero_y_idx[right_idx] else 1-float(abs(c_idx-l_idx))/float(abs(l_idx-r_idx))
					w2=1-w1
					if not (w1 >=0 and w2>=0 and w1<=1 and w2 <=1):
						import ipdb
						ipdb.set_trace()
					assert(w1 >=0 and w2>=0 and w1<=1 and w2 <=1)
					all_Y[c_idx]=w1*list_Y[l_idx] +w2*list_Y[r_idx]
					# all_Y[c_idx]=w1*(list_Y[l_idx] +w2*list_Y[r_idx])/2.0
		return all_Y
		# ------------------------------------------------------------------
	# def prepare_test_data(self,mat):
	# 	# interpolte (zero fills to 2 hours)
	# 	h,w=mat.shape
	# 	X_test_hour  = 2
	# 	X_test_n     =   int(math.floor(X_train_hourse*60/self.time_interval))
	# 	test_x =np.zeros()

	def delete_zero_y_data(self,X_list,Y_list,route_id_idx):
		# for i,y_value in enumerate(y):
		# 	zero_y_index=[i for i, y_v in enumerate(y) if np.prod(y_v)==0]
		assert(len(Y_list)==len(X_list) or len(Y_list) ==1)
		Y=Y_list[route_id_idx] if len(Y_list)==len(X_list) else Y_list[0]
		# zero_y_index=[i for i, y_v in enumerate(Y) if np.prod(y_v)==0]
		zero_y_index=[i for i, y_v in enumerate(Y) if np.sum(y_v)==0]
		len_d =len(X_list)
		for i in range(len_d):
			X_list[i] =np.delete(X_list[i],zero_y_index,axis=0)
			if len(Y_list)==len(X_list) and len(Y_list)>1: 
				Y_list[i] =np.delete(Y_list[i],zero_y_index,axis=0)
		if len(Y_list) ==1:
			Y_list[0] =np.delete(Y_list[0],zero_y_index,axis=0)
		return X_list, Y_list,zero_y_index

	def prepare_train_data(self,time_features,time_stamp,p_hour = 2):
		X_train_hours				=	p_hour
		prediction_hours			=	2
		prediction_interval_minutes =	self.time_interval
		X_predict_n                 =   int(math.floor(X_train_hours*60 / self.time_interval))
		Y_predict_n            		= 	int(math.floor(prediction_hours*60 /prediction_interval_minutes))
		Y_hours_div_by_20min_n      = 	int(math.floor(prediction_hours*60 /20))
		len_time_windows_hours 		= 	int(math.floor((X_train_hours+prediction_hours)*60/prediction_interval_minutes))
		len_f =len(time_features)
		sample_n =int(len_f-len_time_windows_hours+1)
		lx =len(time_features[0])
		# import ipdb
		# ipdb.set_trace()
		X_train =np.zeros((sample_n, int(lx*X_predict_n)))
		# Y_train =np.zeros((sample_n, int(Y_predict_n)))
		Y_train =np.zeros((sample_n, int(Y_hours_div_by_20min_n)))
		sample_weights =[]
		t_start_1=datetime(2000,1,1,8,0,0)
		t_end_1=datetime(2000,1,1,10,0,0)
		Time_span_1=(t_start_1,t_end_1)
		t_start_2=datetime(2000,1,1,17,0,0)
		t_end_2=datetime(2000,1,1,19,0,0)
		Time_span_2=(t_start_2,t_end_2)


		print(len_f)
		all_Y=[]
		# interplote the missed Y (average time)
		# all_Y=[i for i in  ]
		for l in range(len_f):
			all_Y.append(time_features[l][1])
		print('start interpolate Y')
		# all_Y=self.interpolate_zerolist(all_Y)
		t20_min_to_curmin_r =int(Y_predict_n/Y_hours_div_by_20min_n)
		X=[]
		Y_given_interval=[]
		for l in range(sample_n):
			Y_20min_interval=[]
			if l ==0:
				for i in range(X_predict_n):
					X.append(time_features[l+i])
				for y in range(Y_predict_n):
					Y_given_interval.append(all_Y[l+X_predict_n+y])
					
					# import ipdb
					# ipdb.set_trace()
			else:
				X.pop(0)
				X.append(time_features[l+X_predict_n-1])
				Y_given_interval.pop(0)
				Y_given_interval.append(all_Y[l+X_predict_n+Y_predict_n-1])

			
			# equivalent to (20min/given interval minutes(1,5,10 typically)
			for y in range(Y_hours_div_by_20min_n):
				Y_range = Y_given_interval[y*t20_min_to_curmin_r:(y+1)*t20_min_to_curmin_r]
				non_zero_y =[Y_range[i] for i, e in enumerate(Y_range) if e!=0]
				if len(non_zero_y)==0:
					non_zero_y.append(0)
				Y_20min_interval.append(np.mean(non_zero_y))
				# ipdb.set_trace()

				# Y_20min_interval.append(np.mean(Y_given_interval[y*t20_min_to_curmin_r:(y+1)*t20_min_to_curmin_r]))
			last=time_stamp[l+X_predict_n+Y_predict_n-1]
			s_weight=1 if last.time()>=t_start_1.time() and last.time()<t_end_1.time() or \
						last.time()>=t_start_2.time() and last.time()<t_end_2.time() \
						else 0.2
			sample_weights.append(s_weight)

			X_train[l,:]=np.array(X).flatten()
			Y_train[l,:]=np.array(Y_20min_interval)
			# ipdb.set_trace()
		
		# ipdb.set_trace()
		return X_train, Y_train, sample_weights
		# return X_train, Y_train
		# return np.array(X_train), np.array(Y_train)


	def convert_windowInfo_to_vector(self,time_recod,time_start,weather_data):
		V=list(time_recod.car_traj_time.values())
		# print (type(V))
		v_count =len(V)
		if v_count >0:
			time_mean =np.mean(V) #sum(V)/float(v_count)
			time_std  =np.std(V)
			start_times = list(time_recod.car_start_time.values())
			start_day   = start_times[0].weekday()
			start_hour  = start_times[0].hour
			start_minute  = start_times[0].minute
			all_start_minutes =[]
			for t in start_times:
				all_start_minutes.append(t.minute)
			if len(all_start_minutes)>1:
				all_start_minutes.sort()
				start_min_diff =np.diff(all_start_minutes)
				mean_start_min_diff=np.mean(start_min_diff)
				std_start_min_diff=np.std(start_min_diff) 
			else:
				mean_start_min_diff =-1
				std_start_min_diff =-1

		else:
			time_mean =0
			time_std  =0
			start_times =-1
			start_day = 0
			start_hour = 0
			start_minute =-1
			mean_start_min_diff =-1
			std_start_min_diff =-1


		# python 3.x
		# for key, values in  time_recod.link_tt.items():
		# python 2.7
		link_mean =[]
		link_std  =[]
		link_count =[]
		link_ids =list(time_recod.link_tt.keys())
		link_ids.sort()
		
		link_mean_st_seconds =[]
		link_std_st_seconds  =[]

		all_link_stat_tt={}
		all_link_stat_st={}
		for id in self.link_ids:
			all_link_stat_tt[id]=[]
			all_link_stat_st[id]=[]
		l1=len(all_link_stat_tt)
		for id in time_recod.link_tt:
			all_link_stat_tt[id]=time_recod.link_tt[id]
			all_link_stat_st[id]=time_recod.link_st[id]
		l2 =len(all_link_stat_tt)
		# import ipdb
		# ipdb.set_trace()
		assert(l1==l2)

		for id in all_link_stat_tt:
			value=all_link_stat_tt[id]
			link_count.append(len(value))
			link_start_times=list(all_link_stat_st[id])
			seconds = []
			for ldate_time in link_start_times:
				s=timedelta(hours=ldate_time.hour,minutes=ldate_time.minute,seconds=ldate_time.second).seconds
				seconds.append(s)
			if len(value) == 0:
				link_mean.append(0)
				link_std.append(0)
				link_mean_st_seconds.append(-1)
				link_std_st_seconds.append(-1)
			else:
				link_mean.append(np.mean(value))
				link_std.append(np.std(value))
				link_mean_st_seconds.append(np.mean(s))
				link_std_st_seconds.append(np.std(s))
		weather_feature=self.get_weather_data(weather_data,time_start)
		# sec_diff=np.diff(seconds)
		# vector =[v_csecondsount,time_mean,time_std,start_day,start_hour,start_minute,mean_start_min_diff] + link_count+link_mean+link_std
		# vector =[v_count,time_mean,time_std]+self.one_hot(start_day+1,8) +self.one_hot(start_hour+1,25) \
		# 		+self.one_hot(start_minute+1,61)+[mean_start_min_diff] + link_mean_st_seconds+link_std_st_seconds \
		# 		+[std_start_min_diff]+link_count+link_mean+link_std+weather_feature
		vector =[v_count,time_mean,time_std] +one_hot(start_day+1,8) +[start_hour] \
				+[mean_start_min_diff] +link_std_st_seconds \
				+[std_start_min_diff]+link_count+link_mean+link_std+weather_feature
		# ipdb.set_trace()
		# import ipdb
		# ipdb.set_trace()
		# print vector
		return vector

	def format_data_in_timeInterval(self,traj_data,vol_data):
		travel_times={}
		print (len(traj_data))
		for i in range(len(traj_data)):
			each_traj = traj_data[i].replace('"', '').split(',')
			intersection_id = each_traj[0]
			tollgate_id = each_traj[1]
			route_id = intersection_id + '-' + tollgate_id
			if route_id not in travel_times.keys():
				travel_times[route_id] = {}

			trace_start_time = each_traj[3]
			trace_start_time = datetime.strptime(trace_start_time, "%Y-%m-%d %H:%M:%S")
			# time_window_minute = int(math.floor(trace_start_time.minute / self.time_interval) * self.time_interval)
			time_window_minute = int(math.floor(trace_start_time.minute / self.time_interval) * self.time_interval)
			start_time_window = datetime(trace_start_time.year, trace_start_time.month, trace_start_time.day,
										trace_start_time.hour, time_window_minute, 0)
			tt = float(each_traj[-1]) # travel time
			# print i
			if start_time_window not in travel_times[route_id].keys():
				travel_times[route_id][start_time_window] = time_recod_data()
			t_record =travel_times[route_id][start_time_window]
			t_record.car_traj_time[each_traj[2]]=tt # each_traj[2] is CardID!
			t_record.car_start_time[each_traj[2]]=trace_start_time
			# t_record.weather=self.parse_weather_info()
			# print travel_times[route_id][start_time_window].car_traj_time[each_traj[2]]
			link_seq =each_traj[4]
			# print link_seq
			link_seq= link_seq.split(';')
			# print link_seq
			# import ipdb
			for link in link_seq:
				info = link.split('#')
				# print info
				if info[0] not in t_record.link_tt.keys():
					t_record.link_tt[info[0]]=[]
					t_record.link_st[info[0]]=[]
				t_record.link_tt[info[0]].append(float(info[2]))
				t_record.link_st[info[0]].append(datetime.strptime(info[1], "%Y-%m-%d %H:%M:%S"))
			# ipdb.set_trace()

		return travel_times
	def get_traj_raw_records_train(self,r_id):
		time_file = path+'trajectories(table 5)_training'+file_suffix
		vol_file =path+'volume(table 6)_training'+file_suffix
		road_link_file=path+'links (table 3)'+file_suffix 
		weather_file =path +'weather (table 7)_training_update'+file_suffix
		traj_data,vol_data,link_ids_data,weather_data=self.load_data(time_file,vol_file,road_link_file,weather_file)
		self.link_ids=self.parse_road_link_ids(link_ids_data)
		weather_records=self.parse_weather_info(weather_data)
		return traj_data, self.link_ids,weather_records


	def get_traj_record_based_train_data(self,r_id='B-3',
										remove_holiday_data=True,
										time_slots_for_sample_weights=
										((time(8,0),time(10,0)),(time(17,0),time(19,0)))):
		time_file = path+'trajectories(table 5)_training'+file_suffix
		vol_file =path+'volume(table 6)_training'+file_suffix
		road_link_file=path+'links (table 3)'+file_suffix 
		weather_file =path +'weather (table 7)_training_update'+file_suffix
		traj_data,vol_data,link_ids_data,weather_records=self.load_data(time_file,vol_file,road_link_file,weather_file)
		self.link_ids=self.parse_road_link_ids(link_ids_data)
		RBTD=record_based_traj_data(traj_data,self.link_ids)
		X,Y,sample_weights=RBTD.generate_training_data(r_id,
														time_slots_for_sample_weights=time_slots_for_sample_weights,
														remove_holiday_data=remove_holiday_data)
		return X,Y,sample_weights
	def load_test1_raw_data(self):
		test_time_file = test_path+'trajectories(table 5)_test1'+file_suffix
		test_vol_file = test_path+'volume(table 6)_test1'+file_suffix
		road_link_file=path+'links (table 3)'+file_suffix
		weather_file =test_path +'weather (table 7)_test1'+file_suffix
		if not hasattr(self,'traj_test1_data'):
			self.traj_test1_data,self.vol_test1_data,link_ids_data,self.weather_test1_records=self.load_data(test_time_file,test_vol_file,road_link_file,weather_file)
			self.link_ids=self.parse_road_link_ids(link_ids_data)
	def get_traj_record_based_test_data(self,r_id):
		self.load_test1_raw_data()
		RBTD=record_based_traj_data(self.traj_test1_data,self.link_ids)
		# ipdb.set_trace()
		X=RBTD.generate_test_data(r_id)
		return X
	def get_traj_record_based_train_generetor(self,r_id='B-3',
										remove_holiday_data=True,
										time_slots_for_sample_weights=
										((time(8,0),time(10,0)),(time(17,0),time(19,0)))):
		time_file = path+'trajectories(table 5)_training'+file_suffix
		vol_file =path+'volume(table 6)_training'+file_suffix
		road_link_file=path+'links (table 3)'+file_suffix 
		weather_file =path +'weather (table 7)_training_update'+file_suffix
		traj_data,vol_data,link_ids_data,weather_records=self.load_data(time_file,vol_file,road_link_file,weather_file)
		self.link_ids=self.parse_road_link_ids(link_ids_data)
		RBTD=record_based_traj_data(traj_data,self.link_ids,testing=True)
		RBTD.set_train_generator(r_id,time_slots_for_sample_weights=time_slots_for_sample_weights, \
								remove_holiday_data=remove_holiday_data,y_interval=20,p_hours=2)
		return RBTD.train_generator



if __name__ == "__main__":
	ls=test_links_to_image_convertor()



# clf.score(X_train,Y_train)
# autor=autosklearn.regression.AutoSklearnRegressor()
# autor.fit(X_train,Y_train)



	


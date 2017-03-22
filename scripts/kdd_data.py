import math
from datetime import datetime,timedelta
import numpy as np
# import autosklearn.regression
import sklearn.cross_validation
import sklearn.metrics
from sklearn import linear_model
from sklearn.cross_validation import cross_val_predict
file_suffix = '.csv'
path = '../../kdd_cup_data/dataSets/training/'  # set the data directory
test_path ='../../kdd_cup_data/dataSets/testing_phase1/'
class time_recod_data():
	def __init__(self):
		self.link_tt={}
		self.link_st={}
		self.car_traj_time={}
		self.car_start_time={}
	def zero_init(self,link_ids):
		for l_id in link_ids:
			self.link_tt[l_id] =[]
			self.link_st[l_id] =[]
			# self.link_tt(link_ids)
class kdd_data():
	def __init__(self):
		self.time_interval = 1 # 10 minutes interaval
		time_file = path+'trajectories(table 5)_training'+file_suffix
		vol_file =path+'volume(table 6)_training'+file_suffix
		road_link_file=path+'links (table 3)'+file_suffix 
		weathre_file =path +'weather (table 7)_test1'+file_suffix
		traj_data,vol_data,link_ids_data=self.load_data(time_file,vol_file,road_link_file)
		self.link_ids=self.parse_road_link_ids(link_ids_data)
		# self.travel_times=self.format_data_in_timeInterval(traj_data,vol_data)
		# self.travel_times =self.zerofill_missed_time_info(self.travel_times)
	def read_test_data(self):
		test_time_file = test_path+'trajectories(table 5)_test1'+file_suffix
		test_vol_file = test_path+'trajectories(table 5)_test1'+file_suffix
		road_link_file=path+'links (table 3)'+file_suffix
		traj_data,vol_data,link_ids_data=self.load_data(test_time_file,test_vol_file,road_link_file)
		return self.format_data_in_timeInterval(traj_data,vol_data)
	
	def find_time_range(self,two_h_interval_list,start_time):
		for i,time_range in enumerate(two_h_interval_list):
			if time_range[0] <=start_time and time_range[1] >start_time:
				return i , time_range[0]
	def get_test_feature_mat(self,travel_time_info,route_id):
		time_range_day1_am =('2016-10-18 06:00:00','2016-10-18 08:00:00')
		time_range_day1_pm =('2016-10-18 15:00:00','2016-10-18 17:00:00')

		time_range_day2_am =('2016-10-19 06:00:00','2016-10-19 08:00:00')
		time_range_day2_pm =('2016-10-19 15:00:00','2016-10-19 17:00:00')

		time_range_day3_am =('2016-10-20 06:00:00','2016-10-19 08:00:00')
		time_range_day3_pm =('2016-10-20 15:00:00','2016-10-19 17:00:00')

		time_range_day4_am =('2016-10-21 06:00:00','2016-10-22 08:00:00')
		time_range_day4_pm =('2016-10-21 15:00:00','2016-10-22 17:00:00')

		time_range_day5_am =('2016-10-22 06:00:00','2016-10-22 08:00:00')
		time_range_day5_pm =('2016-10-22 15:00:00','2016-10-22 17:00:00')

		time_range_day6_am =('2016-10-23 06:00:00','2016-10-23 08:00:00')
		time_range_day6_pm =('2016-10-23 15:00:00','2016-10-23 17:00:00')

		time_range_day7_am =('2016-10-24 06:00:00','2016-10-24 08:00:00')
		time_range_day7_pm =('2016-10-24 15:00:00','2016-10-24 17:00:00')

		time_list =[time_range_day1_am,time_range_day1_pm,time_range_day2_am,time_range_day2_pm, \
					time_range_day3_am,time_range_day3_pm,time_range_day4_am,time_range_day4_pm, \
					time_range_day5_am,time_range_day5_pm,time_range_day6_am,time_range_day6_pm, \
					time_range_day7_am,time_range_day7_am]

		# datetime.strptime(trace_start_time, "%Y-%m-%d %H:%M:%S")
		two_h_interval =[]
		dstart =[]
		dend=[]
		for time_str in time_list:
			dstart =datetime.strptime(time_str[0], "%Y-%m-%d %H:%M:%S")
			dend   =datetime.strptime(time_str[1], "%Y-%m-%d %H:%M:%S")
			two_h_interval.append((dstart,dend))
		D=travel_time_info[route_id]
		time_windows =list(D.keys())
		time_windows.sort()
		mat = {}
		for t_w in time_windows:
			t_record = D[t_w]
			vect=self.convert_windowInfo_to_vector(t_record)
			start_times =list(t_record.car_start_time.values())
			start_times.sort()
			print (t_w)
			print(start_times[0])
			# start_min=math.floor(start_times[0]/20)*20
			if two_h_interval is None:
				import ipdb
				ipdb.set_trace()
			print (two_h_interval)

			
			i,time_range=self.find_time_range(two_h_interval,start_times[0])
			if time_range not in mat.keys():
				mat[time_range]=[]
			mat[time_range].append(vect)
			# print vect
		return mat

	def zerofill_missed_time_info(self,travel_times,route_id =None):
		# 1. find min start-time and max-end time across all roud_ids
		start_time = datetime(2100,1,1,1,0)
		end_time  =datetime(1900,1,1,1,0)
		if route_id is not None:
			time_windows  =list(travel_times[route_id].keys())
			time_windows.sort()
			start_time = time_windows[0] if start_time > time_windows[0] else start_time
			end_time = time_windows[-1] if end_time < time_windows[-1] else end_time
		else:
			for r_id in travel_times.keys():
				time_windows  =list(travel_times[r_id].keys())
				time_windows.sort()
				start_time = time_windows[0] if start_time > time_windows[0] else start_time
				end_time = time_windows[-1] if end_time < time_windows[-1] else end_time
		current_time = start_time

		# 2 . get all routes link ids
		link_tt_ids={}
		for rd_ids in travel_times.keys():
			first_time =  next(iter(travel_times[rd_ids]))
			link_tt_ids[rd_ids]=list(travel_times[rd_ids][first_time].link_tt.keys())
		added_time =timedelta(minutes=self.time_interval)
		while current_time <= end_time:
			for r_id in travel_times.keys():
				time_windows  =list(travel_times[r_id].keys())
				if current_time not in time_windows:
					t_record =time_recod_data()
					t_record.zero_init(link_tt_ids[rd_ids])
					travel_times[r_id][current_time] =t_record
					print (current_time)
			current_time+=added_time
		return travel_times
	def one_hot(self,num,length):
		x = [0 for k in range(length)]
		x[num]=1
		return x
	def parse_road_link_ids(self,link_ids_data):
		link_ids =[]
		for i in range(len(link_ids_data)):
			each_line = link_ids_data[i].replace('"', '').split(',')
			link_ids.append(each_line[0])
		# print (len(link_ids))
		return link_ids

		
	def load_data(self,time_file,vol_file,road_link_file):
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
		return traj_data, vol_data, link_ids

	def get_feature_matrix(self,travel_times_struct,route_id):
		D=travel_times_struct[route_id]
		# import ipdb
		# ipdb.set_trace()
		time_windows =list(D.keys())
		time_windows.sort()
		mat=[]
		for t_w in time_windows:
			vect=self.convert_windowInfo_to_vector(D[t_w])
			mat.append(vect)
			# print vect
		return mat
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
					w1=0 if c_idx >non_zero_y_idx[right_idx] else 1-float(abs(c_idx-l_idx+1))/float(abs(l_idx-r_idx+1))
					w2=1-w1
					assert(w1 >=0 and w2>=0 and w1<=1 and w2 <=1)
					all_Y[c_idx]=w1*list_Y[l_idx] +w2*list_Y[r_idx]
					# all_Y[c_idx]=w1*(list_Y[l_idx] +w2*list_Y[r_idx])/2.0
		return all_Y
		# ------------------------------------------------------------------

	def prepare_train_data(self,time_features):
		X_train_hourse 				=	2
		prediction_hourse 			=	2
		prediction_interval_minutes =	self.time_interval
		X_predict_n                 =   int(math.floor(X_train_hourse*60/self.time_interval))
		Y_predict_n            		= 	int(math.floor(prediction_hourse*60 /prediction_interval_minutes))
		Y_hours_div_by_20min_n      = 	int(math.floor(prediction_hourse*60 /20))
		len_time_windows_hours 		= 	int(math.floor((X_train_hourse+prediction_hourse)*60/prediction_interval_minutes))
		len_f =len(time_features)
		sample_n =int(len_f-len_time_windows_hours+1)
		lx =len(time_features[0])
		X_train =np.zeros((sample_n, int(lx*X_predict_n)))
		# Y_train =np.zeros((sample_n, int(Y_predict_n)))
		Y_train =np.zeros((sample_n, int(Y_hours_div_by_20min_n)))


		print(len_f)
		all_Y=[]
		# interplote the missed Y (average time)
		# all_Y=[i for i in  ]
		for l in range(len_f):
			all_Y.append(time_features[l][1])
		print('start interpolate Y')
		all_Y=self.interpolate_zerolist(all_Y)
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
				Y_20min_interval.append(np.mean(Y_given_interval[y*t20_min_to_curmin_r:(y+1)*t20_min_to_curmin_r]))
			X_train[l,:]=np.array(X).flatten()
			Y_train[l,:]=np.array(Y_20min_interval)
		return X_train, Y_train
		# return np.array(X_train), np.array(Y_train)


	def convert_windowInfo_to_vector(self,time_recod):
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
			start_day = -1
			start_hour = -1
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
		# vector =[v_count,time_mean,time_std,start_day,start_hour,start_minute,mean_start_min_diff] + link_count+link_mean+link_std
		vector =[v_count,time_mean,time_std]+self.one_hot(start_day+1,8) +self.one_hot(start_hour+1,25) \
				+self.one_hot(start_minute+1,61)+[mean_start_min_diff] + link_mean_st_seconds+link_std_st_seconds \
				+[std_start_min_diff]+link_count+link_mean+link_std
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


# clf.score(X_train,Y_train)
# autor=autosklearn.regression.AutoSklearnRegressor()
# autor.fit(X_train,Y_train)



	

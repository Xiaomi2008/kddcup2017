import math
from datetime import datetime,timedelta
import numpy as np
file_suffix = '.csv'
path = '../../kdd_cup_data/dataSets/training/'  # set the data directory
class time_recod_data():
	def __init__(self):
		self.link_tt={}
		self.car_traj_time={}
class kdd_data():
	def __init__(self):
		self.time_interval = 20 # 10 minutes interaval
		time_file = path+'trajectories(table 5)_training'+file_suffix
		vol_file =path+'volume(table 6)_training'+file_suffix
		weathre_file =path +'weather (table 7)_test1'+file_suffix
		traj_data,vol_data=self.load_data(time_file,vol_file)
		self.travel_times=self.get_features(traj_data,vol_data)
	def load_data(self,time_file,vol_file):
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
		return traj_data, vol_data

	def get_train_data(self,travel_times_struct,route_id):
		D=travel_times_struct[route_id]
		# print type(D)
		# import ipdb
		# ipdb.set_trace()
		time_windows =list(D.keys())
		time_windows.sort()
		mat=[]
		vect=self.convert_windowInfo_to_vector(D[time_windows[0]])
		# print len(vect)
		for t_w in time_windows:
			vect=self.convert_windowInfo_to_vector(D[t_w])
			mat.append(vect)
			# print vect
		return mat
	def convert_windowInfo_to_vector(self,time_recod):
		V=time_recod.car_traj_time.values()
		v_count =len(V)
		ave_time_mean =sum(V)/float(v_count)
		ave_time_std  =np.std(V)
		# python 3.x
		# for key, values in  time_recod.link_tt.items():
		# python 2.7
		link_mean =[]
		link_std  =[]
		link_count =[]
		link_ids =list(time_recod.link_tt.keys())
		link_ids.sort()
		for id in link_ids:
			value=time_recod.link_tt[id]
			link_count.append(len(value))
			link_mean.append(np.mean(value))
			link_std.append(np.std(value))
		vector =[v_count,ave_time_mean,ave_time_std] + link_count+link_mean+link_std
		return vector
		# for key, value in time_recod.link_tt.iteritems():
		# 	v_l =len(value)
		# 	link_mean.append(np.mean(v_l))
		# 	link_std.append(np.std(v_l))




	def get_features(self,traj_data,vol_data):
		travel_times={}
		print len(traj_data)
		for i in range(len(traj_data)):
			each_traj = traj_data[i].replace('"', '').split(',')
			intersection_id = each_traj[0]
			tollgate_id = each_traj[1]
			route_id = intersection_id + '-' + tollgate_id
			if route_id not in travel_times.keys():
				travel_times[route_id] = {}

			trace_start_time = each_traj[3]
			trace_start_time = datetime.strptime(trace_start_time, "%Y-%m-%d %H:%M:%S")
			time_window_minute = int(math.floor(trace_start_time.minute / self.time_interval) * self.time_interval)
			start_time_window = datetime(trace_start_time.year, trace_start_time.month, trace_start_time.day,
				trace_start_time.hour, time_window_minute, 0)
			tt = float(each_traj[-1]) # travel time
			# print i
			if start_time_window not in travel_times[route_id].keys():
				travel_times[route_id][start_time_window] = time_recod_data()
			t_record =travel_times[route_id][start_time_window]
			t_record.car_traj_time[each_traj[2]]=tt
			# print travel_times[route_id][start_time_window].car_traj_time[each_traj[2]]
			link_seq =each_traj[4]
			# print link_seq
			link_seq= link_seq.split(';')
			# print link_seq
			for link in link_seq:
				info = link.split('#')
				# print info
				if info[0] not in t_record.link_tt.keys():
					t_record.link_tt[info[0]]=[]
				t_record.link_tt[info[0]].append(float(info[2]))
		return travel_times

A =kdd_data()

route_time_windows = list(A.travel_times['C-3'].keys())
route_time_windows.sort()
# print route_time_windows[0:60]
mat =A.get_train_data(A.travel_times,'C-3')


	

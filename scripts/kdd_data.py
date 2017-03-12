import math
from datetime import datetime,timedelta
file_suffix = '.csv'
path = '../../kdd_cup_data/dataSets/training/'  # set the data directory
class kdd_data():
	def __init__(self):
		self.time_interval = 10 # 10 minutes interaval
		time_file = path+'trajectories(table 5)_training'+file_suffix
		vol_file =path+'volume(table 6)_training'+file_suffix
		weathre_file =path +'weather (table 7)_test1'+file_suffix
		traj_data,vol_data=self.load_data(time_file,vol_file)
	def load_data(time_file,volume_file):
		# Step 1: Load trajectories
		fr = open(time_file, 'r')
		fr.readline()  # skip the header
		traj_data = fr.readlines()
		fr.close()
		#step 2: load valume
		fr = open(val_file, 'r')
		fr.readline()  # skip the header
		vol_data = fr.readlines()
		fr.close()
		return traj_dataa, vol_data
	def get_features(self,traj_data,vol_data):
		self.traval_tiems={}
		for i in range(len(traj_data)):
			each_traj = traj_data[i].replace('"', '').split(',')
			intersection_id = each_traj[0]
			tollgate_id = each_traj[1]
			route_id = intersection_id + '-' + tollgate_id
        if route_id not in self.traval_times.keys():
            travel_times[route_id] = {}
        trace_start_time = each_traj[3]
        trace_start_time = datetime.strptime(trace_start_time, "%Y-%m-%d %H:%M:%S")
        time_window_minute = int(math.floor(trace_start_time.minute / self.time_interval) * self.time_interval)
        start_time_window = datetime(trace_start_time.year, trace_start_time.month, trace_start_time.day,
                                     trace_start_time.hour, time_window_minute, 0)
       
        tt = float(each_traj[-1]) # travel time

        if start_time_window not in travel_times[route_id].keys():
            travel_times[route_id][start_time_window] = [tt]
        else:
            travel_times[route_id][start_time_window].append(tt)


	def get_intersect_gate_features(self,):

	

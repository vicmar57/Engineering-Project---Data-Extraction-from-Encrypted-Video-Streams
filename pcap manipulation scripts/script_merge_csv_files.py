import csv
import numpy as np
import random


with open("C:/Users/nimrod/Desktop/nimrod/Experiments/exp_22_1_4_parts_rand/window_2/test/all_data.csv", 'a+') as all_data:
	all_data.write("sum_data,std_size,avg_size,std_arr_time,avg_arr_time,num_of_pack,lebel\n")
	for index in range(5):
		print(index)
		if index >= 0:
			f = open("C:/Users/nimrod/Desktop/nimrod/Experiments/exp_22_1_4_parts_rand/window_2/test/part_"+str(index)+"_features.csv", "r")
			reader = csv.reader(f)
			for row in reader:
				all_data.write("%.6f" %float(row[0]) + ",%.6f" % float(row[1])+ ",%.6f" % float(row[2]) + ",%.6f" %float(row[3])+ ",%.6f" % float(row[4])+",%.6f" % float(row[5])+",%.6f" % float(row[6])+ "\n")
		


		

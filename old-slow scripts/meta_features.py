import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from math import ceil
import timeit


time_slice = 10; #in seconds

#pull csv data from videos
csv_data = [0,0,0]
csv_data[0]= pd.read_csv(r"C:\Users\wnp387\Desktop\university\camera captures\hik pcaps n CSVs\0man\hik_0.csv")
csv_data[1]=pd.read_csv(r"C:\Users\wnp387\Desktop\university\camera captures\hik pcaps n CSVs\1man\hik_1.csv")
csv_data[2]=pd.read_csv(r"C:\Users\wnp387\Desktop\university\camera captures\hik pcaps n CSVs\2man\hik_2.csv")

csv_data[0]['Label'] = 0;
csv_data[1]['Label'] = 1;
csv_data[2]['Label'] = 2;

seconds_in_DFs_list = list(range(len(csv_data))) #in a every single video dataframe
time_slice_list = list(range(len(csv_data))) #initiallization
indexes_to_split_frame_by_list = list(range(len(csv_data))) #initiallization
time_sliced_df_list = []

start = timeit.default_timer() #to measure runtime

for i in list(range(len(csv_data))) : #slice data frames by time slice seconds
    #"lose" non essential data
   csv_data[i] = csv_data[i].drop(['Source' , 'Destination' , 'Protocol' , 'Info' , 'No.'], axis=1)
   csv_data[i]['Time']  = csv_data[i]['Time'].astype('float') #change Time column data to int type 
   
   seconds_in_DFs_list[i] = int(ceil(csv_data[i].at[len(csv_data[i])-1,'Time'])) #number of seconds in current df
   time_slice_list[i] = list(range(0,seconds_in_DFs_list[i],time_slice)) #list of times to slice by
   indexes_to_split_frame_by_list[i] = list(range(0,seconds_in_DFs_list[i],time_slice)) #initialization
   
   for j in list(range(len(time_slice_list[i]))): #calc indexes to slice by
       #find first index of csv_data[i]['Time'] the exceeds time_slice_list[i][j]
       indexes_to_split_frame_by_list[i][j] = np.argmax(csv_data[i]['Time'] > time_slice_list[i][j]) -1
            
   df_slice = pd.DataFrame() #TODO: change split into a single df, not list
   
   for j in list(range(len(time_slice_list[i]))):
       #print(len(time_slice_list[i]))
       if (j != len(time_slice_list[i])-1 and j != 0):  #not first or last index
           ind_start = indexes_to_split_frame_by_list[i][j] + 1
           ind_end = indexes_to_split_frame_by_list[i][j+1] + 1
       elif j == len(time_slice_list[i])-1:             #last index
           ind_start = indexes_to_split_frame_by_list[i][j] + 1
           ind_end = len(csv_data[i])
       else:                                            # j==0, first index
           ind_start = 0
           ind_end = indexes_to_split_frame_by_list[i][j+1]+1
       
       df_slice = csv_data[i][ind_start:ind_end]
       df_slice['Time'] = df_slice['Time'].diff().fillna(0) #calc time diffs, not time since begining of recording packets.
       time_sliced_df_list.append(df_slice)
         
       
#create dataframe for meta-features.
index = range(len(time_sliced_df_list))
columns = ['time_diff_std','time_diff_mean', 'packet_size_std','packet_size_mean','packet_size_tot','label']
final_df = pd.DataFrame(index=index, columns=columns)


for i in range(len(time_sliced_df_list)):
    if len(time_sliced_df_list[i]) != 0:
        final_df.iloc[i]["time_diff_std"] = time_sliced_df_list[i]['Time'].std() #Standard deviation of time diffs 
        final_df.iloc[i]["time_diff_mean"] = time_sliced_df_list[i]['Time'].mean() #Average difference between the arrival time of two adjacent packets. 
        final_df.iloc[i]["packet_size_mean"] = time_sliced_df_list[i]['Length'].mean() #average packet size in the window
        final_df.iloc[i]["packet_size_tot"] = time_sliced_df_list[i]['Length'].sum() #total data in window in bytes
        final_df.iloc[i]["packet_size_std"] = time_sliced_df_list[i]['Length'].std() #standard deviation of packet sizes in window
        final_df.iloc[i]["label"] = time_sliced_df_list[i]['Label'].iloc[0] #keep same labels


final_df = final_df[pd.notnull(final_df['time_diff_std'])] #remove nan rows
final_df = final_df.sample(frac=1).reset_index(drop=True) #mix the data!!!

stop = timeit.default_timer()
print('')
print('')
print('')
print('RunTime:', stop - start, 'sec')

labels = final_df['label'] #get labels
final_df = final_df.drop('label',axis=1) #remove labels from data

train_data = final_df[:int(len(final_df)*0.8)] #first 80% of data
train_labels = labels[:int(len(final_df)*0.8)].astype('int') #first 80% of labels
test_data = final_df[int(len(final_df)*0.8):]  #last 20% of data
test_labels = labels[int(len(final_df)*0.8):].astype('int') #last 20% of labels

clf = RandomForestClassifier()
clf.fit(train_data ,train_labels)
pred = clf.predict(test_data)

acc_score = "{:.3f}".format(accuracy_score(test_labels , pred))
mean_absolute_error = "{:.3f}".format(mean_absolute_error(test_labels , pred))

print ('accuracy_score:',acc_score, '.time_slice:', time_slice, 'sec')
print ('mean_absolute_error:', mean_absolute_error, '.time_slice:', time_slice,'sec')

#TODO: AUC ROC curve 













#to read multiple CSVs
#import glob
#import pandas as pd
#
#path =r'data/luftdaten/5331' # use your path
#
#filenames = glob.glob(path + "/*.csv")
#count_files = 0
#dfs = []
#for filename in filenames:
#    if count_files ==0:
#        dfs.append(pd.read_csv(filename, sep=";")) 
#        count_files += 1
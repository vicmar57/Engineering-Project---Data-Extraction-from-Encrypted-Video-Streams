import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from math import ceil
import timeit
from scipy import stats #clear outliers!!!


time_slice = 10; #in seconds
num_labels = 3;
num_dfs_tot = 0;


#to read multiple CSVs automatically
import glob
path = r'C:\Desktop stuff\university\camera captures\hik pcaps n CSVs' 
allFolderPaths = glob.glob(path + "/*")

labels_dfs = [0]*num_labels

for i in range(num_labels):
    labels_dfs[i] = [pd.read_csv(f) for f in glob.glob(allFolderPaths[i] + "/*.csv")]

for i in range(num_labels) :
    num_dfs_tot += len(labels_dfs[i])
    for j in range(len(labels_dfs[i])) :
        # cut 15 sec of start and end of each video.
        labels_dfs[i][j] = labels_dfs[i][j][labels_dfs[i][j].Time > 15]
        labels_dfs[i][j] = labels_dfs[i][j][labels_dfs[i][j].Time < labels_dfs[i][j]["Time"].iloc[-1] - 15]
        labels_dfs[i][j]['Time'] -= 15
        labels_dfs[i][j]['Label'] = i 


# about 4s to read
    
#%% rest

combined = pd.DataFrame()

seconds_in_DFs_list = [0]*num_dfs_tot #in a every single video dataframe
time_slice_list     = [0]*num_dfs_tot #initiallization
indexes_to_split_frame_by_list = [0]*num_dfs_tot #initiallization
time_sliced_df_list = []


start = timeit.default_timer() #to measure runtime

for k in range(num_labels) : #slice data frames by time slice seconds

    for i in list(range(len(labels_dfs[k]))) : #slice data frames by time slice seconds
        #"lose" non essential data
       labels_dfs[k][i] = labels_dfs[k][i].drop(['Source' , 'Destination' , 'Protocol' , 'Info' , 'No.'], axis=1)
       labels_dfs[k][i]['Time']  = labels_dfs[k][i]['Time'].astype('float') #change Time column data to int type 

       combined = combined.append(labels_dfs[k][i])
       
       seconds_in_DFs_list[i] = int(ceil(labels_dfs[k][i].at[len(labels_dfs[k][i])-1,'Time'])) #number of seconds in current df
       time_slice_list[i] = list(range(0,seconds_in_DFs_list[i],time_slice)) #list of times to slice by
       indexes_to_split_frame_by_list[i] = list(range(0,seconds_in_DFs_list[i],time_slice)) #initialization
       
       for j in list(range(len(time_slice_list[i]))): #calc indexes to slice by
           #find first index of labels_dfs[k][i]['Time'] the exceeds time_slice_list[i][j]
           indexes_to_split_frame_by_list[i][j] = np.argmax(labels_dfs[k][i]['Time'] > time_slice_list[i][j]) -1 #TODO: moved below.
                
       df_slice = pd.DataFrame() 
       
       for j in list(range(len(time_slice_list[i]))): #TODO: change for efficiency.
           #print(len(time_slice_list[i]))
           #indexes_to_split_frame_by_list[i][j] = np.argmax(labels_dfs[k][i]['Time'] > time_slice_list[i][j]) -1 TODO: doesnt work because we need j+1.
           if (j != len(time_slice_list[i])-1 and j != 0):  #not first or last index
               ind_start = indexes_to_split_frame_by_list[i][j] + 1
               ind_end = indexes_to_split_frame_by_list[i][j+1] + 1
           elif j == len(time_slice_list[i])-1:             #last index
               ind_start = indexes_to_split_frame_by_list[i][j] + 1
               ind_end = len(labels_dfs[k][i])
           else:                                            # j==0, first index
               ind_start = 0
               ind_end = indexes_to_split_frame_by_list[i][j+1]+1
           
           df_slice = labels_dfs[k][i][ind_start:ind_end]
           df_slice.loc[:,'Time'] = df_slice['Time'].diff().fillna(0) #calc time diffs, not time since begining of recording packets.
           df_slice = df_slice.rename(index=str, columns={"Time": "TimeDiff"}) #change Time column name to time-diff 
           time_sliced_df_list.append(df_slice)
           

combined.to_csv('combined.csv',index=False)
#create dataframe for meta-features.
index = range(len(time_sliced_df_list))
columns = ['time_diff_std','time_diff_mean', 'packet_size_std','packet_size_mean','packet_size_tot','packet_num','label']
final_df = pd.DataFrame(index=index, columns=columns)

stop = timeit.default_timer()
print('\n\nRunTime:', stop - start, 'sec')

time_sliced_df_list = [item for item in time_sliced_df_list if len(item) > 200]
          
for i in range(len(time_sliced_df_list)):
    #if len(time_sliced_df_list[i]) > 200: # more than 200 samples
    final_df.iloc[i]["packet_num"] = time_sliced_df_list[i]['TimeDiff'].count() #number of packets in dataframe
    final_df.iloc[i]["time_diff_std"] = time_sliced_df_list[i]['TimeDiff'].std() #Standard deviation of time diffs 
    final_df.iloc[i]["time_diff_mean"] = time_sliced_df_list[i]['TimeDiff'].mean() #Average difference between the arrival time of two adjacent packets. 
    final_df.iloc[i]["packet_size_mean"] = time_sliced_df_list[i]['Length'].mean() #average packet size in the window
    final_df.iloc[i]["packet_size_tot"] = time_sliced_df_list[i]['Length'].sum() #total data in window in bytes
    final_df.iloc[i]["packet_size_std"] = time_sliced_df_list[i]['Length'].std() #standard deviation of packet sizes in window
    final_df.iloc[i]["label"] = time_sliced_df_list[i]['Label'].iloc[0] #keep same labels


final_df = final_df[pd.notnull(final_df['time_diff_std'])] #remove nan rows

final_df = final_df.infer_objects() #convert all data types in DF to numeric
#print(final_df.dtypes)


final_df.to_csv('meta_features_with_packet_num.csv',index=False) #save to csv


#sample 80% of each label for train, and 20% of each label for test
train_data = []
test_data = []
for i in range(num_labels) :
    single_label_df = final_df[final_df.label == i]
    zscore = np.abs(stats.zscore(single_label_df.drop('label',axis=1))) # clear outliers from every label's DF
    single_label_df = single_label_df[(zscore < 3).all(axis=1)] # 95% of valid data! - 3 STDs from normal distr.
    train_data.append(single_label_df[:int(len(single_label_df)*0.8)]) #first 80% of data
    test_data.append(single_label_df[int(len(single_label_df)*0.8):])  #last 20% of data

train_data  = pd.concat(train_data, axis=0) #first 80% of data for label i
test_data   = pd.concat(test_data, axis=0)   #last 20% of data for label i
    

train_data = train_data.sample(frac=1).reset_index(drop=True) #mix the data!!!
test_data  = test_data.sample(frac=1).reset_index(drop=True) #mix the data!!!
train_labels = train_data['label'] #get labels
test_labels = test_data['label'] #get labels
train_data = train_data.drop('label',axis=1) #remove labels from data
test_data = test_data.drop('label',axis=1) #remove labels from data



#labels = final_df['label'] #get labels
#final_df = final_df.drop('label',axis=1) #remove labels from data
#
#
#
#train_data = final_df[:int(len(final_df)*0.8)] #first 80% of data
#train_labels = labels[:int(len(final_df)*0.8)].astype('int') #first 80% of labels
#test_data = final_df[int(len(final_df)*0.8):]  #last 20% of data
#test_labels = labels[int(len(final_df)*0.8):].astype('int') #last 20% of labels

clf = RandomForestClassifier()
clf.fit(train_data ,train_labels)
pred = clf.predict(test_data)

acc_score = "{:.2f}".format(accuracy_score(test_labels , pred))
mean_absolute_error = "{:.3f}".format(mean_absolute_error(test_labels , pred))

print ('accuracy_score:',acc_score, '.time_slice:', time_slice, 'sec')
print ('mean_absolute_error:', mean_absolute_error, '.time_slice:', time_slice,'sec')

#TODO: AUC ROC curve 

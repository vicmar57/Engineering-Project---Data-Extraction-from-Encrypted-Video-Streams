import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from math import ceil
import timeit

time_slice = 10; #in seconds
num_labels = 3;
num_dfs_tot = 0
labels_dfs = [0]*num_labels

#to read multiple CSVs automatically
import glob
path = r'C:\Desktop stuff\university\camera captures\hik pcaps n CSVs' 
allFolderPaths = glob.glob(path + "/*")


#TODO: read all CSVs for label i, and cut 15 sec of start and end of each video. remove outliers.
for i in range(num_labels) :
    labels_dfs[i] = [pd.read_csv(f) for f in glob.glob(allFolderPaths[i] + "/*.csv")]
    num_dfs_tot += len(labels_dfs[i])

    for j in range(len(labels_dfs[i])) :
        labels_dfs[i][j] = labels_dfs[i][j][labels_dfs[i][j].Time > 15]
        labels_dfs[i][j] = labels_dfs[i][j][labels_dfs[i][j].Time < labels_dfs[i][j]["Time"].iloc[-1] - 15]
        labels_dfs[i][j]['Time'] -= 15
        labels_dfs[i][j]['Label'] = i

#%% Find the Time diff that shows when a frame ends

combined = pd.DataFrame()

seconds_in_DFs_list = [0]*num_dfs_tot #in a every single video dataframe
time_slice_list     = [0]*num_dfs_tot #initiallization
indexes_to_split_frame_by_list = [0]*num_dfs_tot #initiallization
time_sliced_df_list = []


start = timeit.default_timer() #to measure runtime

for k in range(num_labels) : #slice data frames by time slice seconds
    for i in list(range(len(labels_dfs[k]))) : #slice data frames by time slice seconds
        #"lose" non essential data
       labels_dfs[k][i].drop(['Source' , 'Destination' , 'Protocol' , 'Info' , 'No.'], axis=1, inplace=True)
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
           time_sliced_df_list.append(df_slice)
         

combined.to_csv('combined.csv',index=False)

#%%

timeDiffCol = labels_dfs[1][1]['Time']
import matplotlib.pyplot as plt
plt.figure
plt.hist(timeDiffCol, bins = 30)
plt.xlabel('timeDiff', fontsize=16); plt.ylabel('count', fontsize=16)


#%% Distinguish between the frames

labels_dfs[1][1]['frameNum'] = 0
j=0
timeDiffCol.reset_index(drop = True , inplace = True)
labels_dfs[1][1].reset_index(drop = True ,inplace = True)

#for i in range(len(timeDiffCol)) :
#   labels_dfs[1][1]['frameNum'] = j
#   if(timeDiffCol[i] >= 0.025) : 
#       j += 1

#labels_dfs[1][1].iterrows()

import time
t0 = time.time()
for idx,row in labels_dfs[1][1].iterrows():
#    labels_dfs[1][1]['frameNum'] = j
    if(timeDiffCol[idx] < 0.025) : 
        labels_dfs[1][1].at[idx,'frameNum'] = j
    else:
        j += 1
        labels_dfs[1][1].at[idx,'frameNum'] = j
#        print(j)
print ('time:  ', time.time()-t0)


united = labels_dfs[1][1].groupby('frameNum',axis=0).sum()
united['Time'] = np.cumsum(united['Time'])

plt.plot(united['Time'],united['Length'])

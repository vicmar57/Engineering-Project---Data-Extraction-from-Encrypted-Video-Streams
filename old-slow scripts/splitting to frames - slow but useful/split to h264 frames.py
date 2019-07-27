# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 12:32:30 2019

@author: Tsur
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from math import ceil
import timeit
from skimage import filters
import matplotlib.pyplot as plt

num_labels = 3;
num_dfs_tot = 0
labels_dfs = [0]*num_labels
thresholds = [0]*num_labels
combined_DF= pd.DataFrame()

#to read multiple CSVs automatically
import glob
path = r'C:\Users\Tsur\Desktop\uni\project\python scripts\gitbuck\csvs' 
allFolderPaths = glob.glob(path + "/*")

#%%
timeDiffCol = pd.Series()
n=0

for i in range(num_labels) :
    labels_dfs[i] = [pd.read_csv(f) for f in glob.glob(allFolderPaths[i] + "/*.csv")]
    num_dfs_tot += len(labels_dfs[i])
    thresholds[i] = [0]*len(labels_dfs[i])
    
    for j in range(len(labels_dfs[i])) :
        labels_dfs[i][j].drop(['Source' , 'Destination' , 'Protocol' , 'Info' , 'No.'], axis=1, inplace=True)
        labels_dfs[i][j] = labels_dfs[i][j][labels_dfs[i][j].Time > 15]
        labels_dfs[i][j] = labels_dfs[i][j][labels_dfs[i][j].Time < labels_dfs[i][j]["Time"].iloc[-1] - 15]
        labels_dfs[i][j]['Time'] -= 15
        labels_dfs[i][j]['Label'] = i
        labels_dfs[i][j].loc[:,'Time'] = labels_dfs[i][j]['Time'].diff().fillna(0) #calc time diffs, not time since begining of recording packets.
        thresholds[i][j] = filters.threshold_otsu(labels_dfs[i][j]['Time'],nbins=30)
        df_size = len(labels_dfs[i][j])
        
        for idx,row in labels_dfs[i][j].iterrows():

            if(labels_dfs[i][j].at[idx,'Time'] < thresholds[i][j]) : 
                labels_dfs[i][j].at[idx,'frameNum'] = n
            else:
                n += 1
                labels_dfs[i][j].at[idx,'frameNum'] = n
                labels_dfs[i][j].at[idx,'Time diff pre'] = labels_dfs[i][j].at[idx,'Time']
        
        
        combined_DF = pd.concat([combined_DF , labels_dfs[i][j]])

def std(x) : return np.std(x)
united = combined_DF.groupby('frameNum' ,sort = 'False').agg({'Time': ['std' ,'mean'],
                                                                  'Length' : ['std' , 'mean' , 'sum'],
                                                                  'Label' : 'first',
                                                                  'Time diff pre':'first'}).fillna(0)
    
        
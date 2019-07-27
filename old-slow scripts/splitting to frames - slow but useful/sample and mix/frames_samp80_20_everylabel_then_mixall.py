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
path = r'C:\Desktop stuff\university\camera captures\hik pcaps n CSVs' 
allFolderPaths = glob.glob(path + "/*")


timeDiffCol = pd.Series()
n=0
start = timeit.default_timer() #to measure runtime


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
                labels_dfs[i][j].at[idx,'lenSlopeDiff'] = labels_dfs[i][j].at[idx,'Length']-labels_dfs[i][j].at[idx-1,'Length']
                labels_dfs[i][j].at[idx,'lenSlopeRel'] = labels_dfs[i][j].at[idx,'Length']/labels_dfs[i][j].at[idx-1,'Length']
                labels_dfs[i][j].at[idx-1,'Time diff next'] = labels_dfs[i][j].at[idx,'Time']

        
        combined_DF = pd.concat([combined_DF , labels_dfs[i][j]])

def std(x) : return np.std(x)
united = combined_DF.groupby('frameNum' ,sort = 'False').agg({'Time': ['std' ,'mean'],
                                                                  'Length' : ['std' , 'mean' , 'sum'],
                                                                  'Label' : 'first',
                                                                  'Time diff pre':'first',
                                                                  'Time diff next' : 'last',
                                                                  'lenSlopeDiff' : 'first',
                                                                  'lenSlopeRel' : 'first'}).fillna(0)
#united = united.sample(frac=1, random_state =1).reset_index(drop=True) #mix the data!!!

#united = united.infer_objects() #convert all data types in DF to numeric

label0 = united[united['Label','first'] == 0]
label1 = united[united['Label','first'] == 1]
label2 = united[united['Label','first'] == 2]

len_data = len(united)

per_0_in_data = len(label0)/len_data
per_1_in_data = len(label1)/len_data
per_2_in_data = len(label2)/len_data


p20_0 = label0[0:(int)(0.2*len_data*per_0_in_data)]
p20_1 = label1[0:(int)(0.2*len_data*per_1_in_data)]
p20_2 = label2[0:(int)(0.2*len_data*per_2_in_data)]

p80_0 = label0[(int)(0.2*len_data*per_0_in_data) : ]
p80_1 = label1[(int)(0.2*len_data*per_1_in_data) : ]
p80_2 = label2[(int)(0.2*len_data*per_2_in_data) : ]

#%%
train_labels  = pd.DataFrame()
test_labels  = pd.DataFrame()
train_data = pd.DataFrame()
test_data = pd.DataFrame()


#print('\nlen of united with I frames:', len(united))
#united.drop(np.where((united['Length','sum']>13000) == True)[0], axis=0, inplace=True)
#united.reset_index(drop=True)
#print('len of united withot I frames:', len(united))

j=0
for i in range(num_labels) :
    label_len = len(np.where(united['Label'] == i)[0])
    mixed_label = united[j:j+label_len].reset_index(drop=True)
    #.sample(frac=1, random_state =1).reset_index(drop=True) #mix the data!!!
    print('len mixed label', len(mixed_label))

    print('len',i,label_len)

#    train_labels = pd.concat([train_labels,mixed_label[0: (int)(label_len*0.8)]['Label']], axis=0) #first 80% of data for label i
    train_data = pd.concat([train_data,mixed_label[0: (int)(label_len*0.8)]], axis=0) #first 80% of data for label i
#    test_labels = pd.concat([test_labels,mixed_label[(int)(label_len*0.8):]['Label']], axis=0) #first 80% of data for label i
    test_data = pd.concat([test_data,mixed_label[(int)(label_len*0.8):]], axis=0) #first 80% of data for label i
    print('len of 20 per mixed_label', len(mixed_label[(int)(label_len*0.8):label_len]) ,(int)(label_len*0.2) )
    print('j:', j)

    j+= label_len+1
    
#train_data = []
#test_data = []
#for i in range(num_labels) :
#    single_label_df = final_df[final_df.label == i]
#    single_label_df = single_label_df.sample(frac=1).reset_index(drop=True) #mix the data!!!
#    train_data.append(single_label_df[:int(len(single_label_df)*0.8)]) #first 80% of data
#    test_data.append(single_label_df[int(len(single_label_df)*0.8):])  #last 20% of data

#united = united.sample(frac=1, random_state =1).reset_index(drop=True) #mix the data!!!



train_data = train_data.sample(frac=1, random_state =1).reset_index(drop=True) #mix the data!!!
test_data = test_data.sample(frac=1, random_state =1).reset_index(drop=True) #mix the data!!!
train_labels = train_data['Label']
test_labels = test_data['Label']


#train_data  = pd.concat([united[:int(len(united)*0.8)]], axis=0) #first 80% of data for label i
#test_data   = pd.concat([united[int(len(united)*0.8):]], axis=0)   #last 20% of data for label i
    
#%%
#train_data = train_data.sample(frac=1).reset_index(drop=True) #mix the data!!!
#test_data  = test_data.sample(frac=1).reset_index(drop=True) #mix the data!!!
#train_labels = train_data['Label'] #get labels
#test_labels = test_data['Label'] #get labels
train_data.drop('Label',axis=1, inplace=True) #remove labels from data
test_data.drop('Label',axis=1, inplace=True) #remove labels from data





#from scipy import stats #clear outliers!!!
#zscore = np.abs(stats.zscore(final_df.drop('label',axis=1))) # clear outliers from every label's DF
#final_df = final_df[(zscore < 3).all(axis=1)] # 95% of valid data! - 3 STDs from normal distr.
#
#
#labels = final_df['label'] #get labels
#final_df = final_df.drop('label',axis=1) #remove labels from data
#
#train_data = final_df[:int(len(final_df)*0.8)] #first 80% of data
#train_labels = labels[:int(len(final_df)*0.8)].astype('int') #first 80% of labels
#test_data = final_df[int(len(final_df)*0.8):]  #last 20% of data
#test_labels = labels[int(len(final_df)*0.8):].astype('int') #last 20% of labels

clf = RandomForestClassifier()
clf.fit(train_data ,train_labels)
pred = clf.predict(test_data)

acc_score = "{:.2f}".format(accuracy_score(test_labels , pred))
mean_absolute_err = "{:.3f}".format(mean_absolute_error(test_labels , pred))

print ('accuracy_score:',acc_score) #, '.time_slice:', time_slice, 'sec')
print ('mean_absolute_error:', mean_absolute_err) #, '.time_slice:', time_slice,'sec')

#TODO: AUC ROC curve 

stop = timeit.default_timer()
print('\n\nRunTime:', stop - start, 'sec')

#%%

import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(true_class, prediction,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(true_class, prediction) # HERE YA VICTOR
    classes = (np.sort(np.unique(true_class)))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

   # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.show()
    
    return


plt.figure(1)
plot_confusion_matrix(test_labels, pred)

#%%

features = list(train_data.columns.values)
importances = clf.feature_importances_
indices = np.argsort(importances)

plt.figure(2)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
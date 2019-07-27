import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from math import ceil
import timeit




#%%

import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(true_class, prediction,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.YlOrRd):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(true_class, prediction) # HERE YA VICTOR
    classes = (np.sort(np.unique(true_class)))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
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

#%%



start = timeit.default_timer() #to measure runtime

time_slice = 2; #in seconds
num_labels = 4;
num_dfs_tot = 0
labels_dfs = [0]*num_labels
sec_to_cut = 15

#to read multiple CSVs automatically
import glob
path = r'C:\Desktop stuff\university\camera captures\hik pcaps n CSVs' 
allFolderPaths = glob.glob(path + "/*")


time_sliced = pd.DataFrame()

#read all CSVs for label i, and cut 15 sec of start and end of each video.
for i in range(num_labels) :
    labels_dfs[i] = [pd.read_csv(f) for f in glob.glob(allFolderPaths[i] + "/*.csv")]
    num_dfs_tot += len(labels_dfs[i])
    
    for j in range(len(labels_dfs[i])) :
        labels_dfs[i][j].drop(['Source' , 'Destination' , 'Protocol' , 'Info' , 'No.'], axis=1, inplace=True)
        labels_dfs[i][j] = labels_dfs[i][j][labels_dfs[i][j].Time > sec_to_cut]
        labels_dfs[i][j] = labels_dfs[i][j][labels_dfs[i][j].Time < labels_dfs[i][j]["Time"].iloc[-1] - sec_to_cut]
        labels_dfs[i][j]['Time'] -= sec_to_cut
        labels_dfs[i][j]['Label'] = i
        labels_dfs[i][j]['vidNum'] = j
        
        df = labels_dfs[i][j] 
        df['TimeSlice'] = (np.array(labels_dfs[i][j]['Time'])/time_slice).astype(int)
        df = df.groupby('TimeSlice',axis=0 ,sort = 'False').agg(
        {'Time'     : ['count', 'std', 'mean'],
         'Length'   : ['mean', 'sum', 'std'],
         'Label'    : 'first',
         'vidNum'    : 'first'}).fillna(0) 
        time_sliced = time_sliced.append(df)

time_sliced.columns = ["_".join(x) for x in df.columns.ravel()]

twenty_per_in_dfs = round(0.2*(time_sliced[time_sliced['Label_first'] == 
                                           0]['vidNum_first'].iloc[-1] +1)).astype(int)

train_data = []
test_data = []
    
train_data.append(time_sliced[time_sliced['vidNum_first'] == 0]) #first 80% of data
train_data.append(time_sliced[time_sliced['vidNum_first'] == 1]) #first 80% of data
test_data.append(time_sliced[time_sliced['vidNum_first'] == 2])  #last 20% of data

train_data  = pd.concat(train_data, axis=0) #first 80% of data for label i
test_data   = pd.concat(test_data, axis=0)   #last 20% of data for label i
train_data.drop('vidNum_first', axis=1, inplace=True)
test_data.drop('vidNum_first', axis=1, inplace=True)

train_labels = train_data['Label_first'] #get labels
test_labels = test_data['Label_first'] #get labels
train_data = train_data.drop('Label_first',axis=1) #remove labels from data
test_data = test_data.drop('Label_first',axis=1) #remove labels from data    
    

clf = RandomForestClassifier(n_estimators = 100)
clf.fit(train_data ,train_labels)
pred = clf.predict(test_data)

acc_score = "{:.2f}".format(accuracy_score(test_labels , pred))
mean_absolute_err = "{:.3f}".format(mean_absolute_error(test_labels , pred))
stop = timeit.default_timer() #to measure runtime
print('\nRunTime:', "{:.1f}".format(stop - start), 'sec ,slice size:', time_slice, 'sec.')
print ('acc_score:',int(float(acc_score)*100), '%, mean_abs_err:', mean_absolute_err)


#%% plotting feature importance and confusion matrix

features = list(train_data.columns.values)
importances = clf.feature_importances_
indices = np.argsort(importances)

plt.figure(2)
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

plt.figure(1)
plot_confusion_matrix(test_labels, pred)

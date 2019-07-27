# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 12:00:03 2018

@author: Tsur
"""

#todo filter tcp packages FROM the camera


import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from math import ceil
training = [0,0,0,0,0,0]

training[0]= pd.read_csv(r"C:\Users\Tsur\Desktop\uni\project\Camera Captures\1 man\1.1.csv")
training[1]=pd.read_csv(r"C:\Users\Tsur\Desktop\uni\project\Camera Captures\2man\2.1.csv")
training[2]=pd.read_csv(r"C:\Users\Tsur\Desktop\uni\project\Camera Captures\1 man\1.2.csv")
training[3]=pd.read_csv(r"C:\Users\Tsur\Desktop\uni\project\Camera Captures\2man\2.2.csv")


training[4]= pd.read_csv(r"C:\Users\Tsur\Desktop\uni\project\Camera Captures\1 man\1.3.csv")
training[5]=pd.read_csv(r"C:\Users\Tsur\Desktop\uni\project\Camera Captures\2man\2.3.csv")


labels = [[ 1 ] , [ 2 ]]
for i in range(len(training)) :
   training[i] = training[i].drop(['Source' , 'Destination' , 'Protocol' , 'Info' , 'No.'], axis=1)
   training[i]['Time']  = training[i]['Time'].astype('float') 
   training[i]['Label'] = i+1;
   
for i in range(len(test)) :
   test[i] = test[i].drop(['Source' , 'Destination' , 'Protocol' , 'Info' , 'No.'], axis=1)
   test[i]['Label'] = i+1;

   
dfmat=pd.concat(training)   
testmat=pd.concat(test)   

print  ceil( training[i].at[len(training[i])-1,'Time']) #WE finished HERE!!!!!!!!!!!!!!!


for i in range  (ceil( training[i].at[len(training[i])-1,'Time'])):
    dfmat['Time'] < i == True

print  training[0].mean(axis=0)


X=dfmat[['Time','Length']]
clf = RandomForestClassifier()
clf.fit(X ,dfmat['Label'])
ans = clf.predict(testmat[['Time','Length']])
print accuracy_score(testmat['Label'],ans)




#for j in range(len(training)) :   
#    for i in range(len(training[j]['Time'])):   
#        if i !=0 :
          #  training[j].loc[i,('Time')] = training[j].loc[i,('Time')] - training[j].loc[i-1,('Time')]
           # training[1]['Time'][i]= training[1]['Time'][i] - training[1]['Time'][i-1]
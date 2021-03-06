import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from math import ceil

time_slice = 3; #in seconds

training = [0,0,0,0,0,0]


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

training[0]= pd.read_csv(r"C:\Users\wnp387\Desktop\university\camera captures\1 man\1.1.csv")
training[1]=pd.read_csv(r"C:\Users\wnp387\Desktop\university\camera captures\2man\2.1.csv")
training[2]=pd.read_csv(r"C:\Users\wnp387\Desktop\university\camera captures\1 man\1.2.csv")
training[3]=pd.read_csv(r"C:\Users\wnp387\Desktop\university\camera captures\2man\2.2.csv")
training[4]= pd.read_csv(r"C:\Users\wnp387\Desktop\university\camera captures\1 man\1.3.csv")
training[5]=pd.read_csv(r"C:\Users\wnp387\Desktop\university\camera captures\2man\2.3.csv")


seconds_in_DF = list(range(len(training))) #in a single video dataframe
time_slice_list = list(range(len(training))) #initiallization
diffList = list(range(len(training))) #initiallization
DFind = 0
newDF = []


training[0]['Label'] = 1;
training[1]['Label'] = 2;
training[2]['Label'] = 1;
training[3]['Label'] = 2;
training[4]['Label'] = 1;
training[5]['Label'] = 2;



for i in list(range(len(training))) : 
   training[i] = training[i].drop(['Source' , 'Destination' , 'Protocol' , 'Info' , 'No.'], axis=1)
   training[i]['Time']  = training[i]['Time'].astype('float') 
   
   seconds_in_DF[i] = int(ceil(training[i].at[len(training[i])-1,'Time']))
   time_slice_list[i] = list(range(0,seconds_in_DF[i],time_slice))
   diffList[i] = list(range(0,seconds_in_DF[i],time_slice))
    
   for j in list(range(len(time_slice_list[i]))):
       diffList[i][j] = np.argmax(training[i]['Time'] > time_slice_list[i][j])
       if j != 0:
           diffList[i][j] -= 1
            
   split = list(range(len(time_slice_list[i])))
   for j in list(range(len(time_slice_list[i]))):
       #print(len(time_slice_list[i]))
       if j == len(time_slice_list[i])-1:
           startInd = diffList[i][j] + 1
           split[j] = training[i][startInd:]
       elif j == 0:
           startInd = 0
           endInd = diffList[i][j+1]+1
           split[j] = training[i][startInd:endInd] 
       else:
           startInd = diffList[i][j] + 1
           endInd = diffList[i][j+1] + 1
           split[j] = training[i][startInd:endInd]   
       newDF.append(split[j])
       
#dataframe for meta-features.
index = range(len(newDF))
columns = ['time_std','time_mean', 'packet_size_std','packet_size_mean','packet_size_tot','label']
final_df = pd.DataFrame(index=index, columns=columns)


for i in range(len(newDF)):
    if len(newDF[i]) != 0:
#        print(i)
#        print(newDF[i]['Time'].std())
        
        final_df.iloc[i]["time_std"] = newDF[i]['Time'].std()-time_slice*i #Standard deviation of difference between the 
        #arrival time of two adjacent packets.
        final_df.iloc[i]["time_mean"] = newDF[i]['Time'].mean()-time_slice*i #Average difference between the arrival time
        #of two adjacent packets. 
        final_df.iloc[i]["packet_size_mean"] = newDF[i]['Length'].mean() #average packet size in the window
        final_df.iloc[i]["packet_size_tot"] = newDF[i]['Length'].sum() #total data in window in bytes
        final_df.iloc[i]["packet_size_std"] = newDF[i]['Length'].std() #standard deviation of packet sizes in window
       #if newDF[i]['Label'].iloc[0] != 'nan:
        final_df.iloc[i]["label"] = newDF[i]['Label'].iloc[0] #keep same labels


final_df = final_df[pd.notnull(final_df['time_std'])] #remove nan rows
        
#final_df = final_df.drop(index=final_df.isnull())
labels = final_df['label']
final_df = final_df.drop('label',axis=1)


train = final_df[:int(len(final_df)*0.8)]
test = final_df[int(len(final_df)*0.8):]


y = labels[:int(len(final_df)*0.8)]
y=y.astype('int')

clf = RandomForestClassifier()
clf.fit(train ,y )
pred = clf.predict(test)
print (accuracy_score(labels[int(len(final_df)*0.8):].astype('int') , pred))
    
    #drop empty dataframes!!!!!!!!!


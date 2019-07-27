import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import sys


#def plot_bitrate(filepath):

#to read multiple CSVs automatically
path = r'C:\Users\wnp387\Desktop\2019-05-07\motion increase test.csv' 
#filename = path.replace('C:\\Users\\wnp387\\Desktop\\2019-05-03\\', '')
#filename = filename.replace('.csv', '')

labels_dfs = pd.read_csv(path)
plt.figure()
#plt.ylim((0,250))
legendTitles = []

labels_dfs.drop(['Source' , 'Destination' , 'Protocol' , 'Info' , 'No.'], axis=1, inplace=True)
#labels_dfs = labels_dfs[labels_dfs.Time > 15]
#labels_dfs = labels_dfs[labels_dfs.Time < labels_dfs["Time"].iloc[-1] - 15]
#labels_dfs['Time'] -= 15

df = labels_dfs
df['key'] = np.array(df['Time']).astype(int)
df = df.groupby('key',axis=0).sum()

df['bytes'] = df['Length'].cumsum()  
#df['BPS'] = df['bytes']/df['Time']
avg_bps = "{:.1f}".format(df['Length'].sum() / (len(df['Length'])*1000))
print("average bitrate:" ,avg_bps)
df['rollLen'] = df['Length'].rolling(20).mean() #in bytes
df['rollLen'] = df['rollLen']/1000 #KBPS
plt.plot(range(len(df['rollLen'])), df['rollLen'])
title = "Bitrate (KBPS) vs time - average: " + avg_bps + " KB/S"
plt.title(title, fontsize=18)
plt.xlabel('Time (sec)', fontsize=18); plt.ylabel('KB', fontsize=16)
    
#filename = path.replace(str('C:\Users\wnp387\Desktop\2019-05-03'), '')
#plt.legend(filename)
plt.show()
    
    
#def hello(name):
#    """Print "Hello " and a name and return None"""
#    print("Hello", name)
#    
#    
#input_filepath = str(sys.argv[1])
#print("input_filepath: " ,input_filepath)
#plot_bitrate(input_filepath)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean 

num_labels = 4;
labels_dfs = [0]*num_labels
fileNames = [0]*num_labels

categories_or_singles = "cat"

#to read multiple CSVs automatically
import glob
path = r'C:\Desktop stuff\university\camera captures\hik pcaps n CSVs' 
allFolderPaths = glob.glob(path + "/*")

plt.figure()
legendTitles = []
ax = plt.gca() 
#max_spec_bitrate = [512]*1200
max_observed_bitrate = [458]*1200
avgs = [0]*num_labels

for i in range(num_labels) :
    labels_dfs[i] = [pd.read_csv(f) for f in glob.glob(allFolderPaths[i] + "/*.csv")]
    fileNames[i] = [f for f in glob.glob(allFolderPaths[i] + "/*.csv")]
    plt.xlim((0,1200))
    color = next(ax._get_lines.prop_cycler)['color']
    avgs[i] = []

#    legendTitles = []

    for j in range(len(labels_dfs[i])) :
        labels_dfs[i][j].drop(['Source' , 'Destination' , 'Protocol' , 'Info' , 'No.'], axis=1, inplace=True)
        labels_dfs[i][j] = labels_dfs[i][j][labels_dfs[i][j].Time > 15]
        labels_dfs[i][j] = labels_dfs[i][j][labels_dfs[i][j].Time < labels_dfs[i][j]["Time"].iloc[-1] - 15]
        labels_dfs[i][j]['Time'] -= 15
        labels_dfs[i][j]['Label'] = i
        labels_dfs[i][j]['vidNum'] = j
        
        short_fileName = fileNames[i][j].replace('C:\\Desktop stuff\\university\\camera captures\\hik pcaps n CSVs\\' 
                                  + str(i) + ' men\\', '')
        short_fileName = short_fileName.replace('.csv', '')
        legendTitles.append(short_fileName) #'vid ' + str(i) + '.' + str(j+1))
        df = labels_dfs[i][j]
        df['key'] = np.array(df['Time']).astype(int)
        df = df.groupby('key',axis=0).sum()
        
        #VR_funcs.PlotXYZ(df['Time'], df['Length'], [1]*len(df['Time']))
        df['bytes'] = df['Length'].cumsum()  
        #df['timeSUM'] = df['Time'].cumsum()  
        df['BPS'] = df['bytes']/df['Time']
        bps = float("{:.1f}".format(df['Length'].sum() / (len(df['Length'])*1000)))
#        df['Length'].sum() / df['Time'].sum()
#        vidNum = df['vidNum'][0]
        avgs[i].append(bps)
        
        df['rollLen'] = df['Length'].rolling(20).mean()
        df['rollLen'] = df['rollLen']/1000
        
        if(categories_or_singles == "cat"):
            if j == 2:
                plt.plot(range(len(df['Length'])), df['rollLen'] , linestyle = '-', color = color, linewidth=4.0)
            else:    
                plt.plot(range(len(df['Length'])), df['rollLen'] , color = color)
        else:
            if j == 2:
                plt.plot(range(len(df['Length'])), df['rollLen'], linestyle = '-', linewidth=4.0)
            else:  
                plt.plot(range(len(df['Length'])), df['rollLen'])
#            plt.ylim((0,350))
        
        #legend = plt.legend(handles=[one, two, three], title="title", loc=4, fontsize='small', fancybox=True)
        plt.xlabel('Time (sec)', fontsize=16); plt.ylabel('KB', fontsize=16)
#        break
        

    avgs[i] = int(mean(avgs[i]))
    avgLine = [avgs[i]]*1200
    legendTitles.append("avg, label = " + str(i))
    plt.plot(range(len(avgLine)), avgLine)
#legendTitles.append("max_spec_bitrate") #'vid ' + str(i) + '.' + str(j+1))    
#plt.plot(range(len(max_spec_bitrate)), max_spec_bitrate)
#legendTitles.append("max_observed_bitrate") #'vid ' + str(i) + '.' + str(j+1))    
#plt.plot(range(len(max_observed_bitrate)), max_observed_bitrate)
plt.title("Videos' bitrates", fontsize=18)
plt.legend(legendTitles)    
plt.show()



# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 20:02:56 2019

@author: WNP387
"""
import numpy as np
import pandas as pd


path = r'C:\Users\wnp387\Desktop\final_pres_vid.csv' 
preds = pd.read_csv(path)
time_slice = 2; #in seconds

preds.drop(['Source' , 'Destination' , 'Protocol' , 'Info' , 'No.'], axis=1, inplace=True)
        
        
preds['TimeSlice'] = (np.array(preds['Time'])/time_slice).astype(int)
preds = preds.groupby('TimeSlice',axis=0 ,sort = 'False').agg(
{'Time'     : ['count', 'std', 'mean'],
 'Length'   : ['mean', 'sum', 'std']}).fillna(0) 
# 'Label'    : 'first',
# 'vidNum'    : 'first'}

preds.columns = ["_".join(x) for x in preds.columns.ravel()]
preds.to_csv('final_pred_df.csv',index=False)

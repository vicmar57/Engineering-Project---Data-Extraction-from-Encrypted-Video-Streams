import cv2
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc
import pandas as pd
import time

width = 860
height = 150
FPS = 24
seconds = 10
x_start = 20

path = r'C:\Desktop stuff\university\camera captures\fast AF test.csv' 
preds = pd.read_csv(path)['Label_first']
preds = preds.sample(frac=1, random_state =1).reset_index(drop=True) #mix the data!!!
preds = preds.loc[:9] #10 2-sec predictions
num_two_secs = len(preds)
time_slice = 2

fourcc = VideoWriter_fourcc(*'MP42')
video = VideoWriter('./pred.avi', fourcc, float(FPS), (width, height))

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (x_start,100)
fontScale              = 3.5
fontColor              = (255,255,255) #white
lineType               = 10            #boldness

for ind in range(num_two_secs): #24 frames per sec - 10 sec
    frame = np.ones((height, width,3), np.uint8) 

    for frame_ind in range(FPS*time_slice): #write time_slice seconds of frames to video
        cv2.putText(frame,"pred: " + str(int(preds[ind])) + " people", 
            bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
        cv2.rectangle(frame, (10,10),(width-10,height-10),(0,0,255),3) #(0,255,0) means all green
        #cv2.imshow("img",frame)
        video.write(frame)
    #time.sleep(time_slice)

video.release()

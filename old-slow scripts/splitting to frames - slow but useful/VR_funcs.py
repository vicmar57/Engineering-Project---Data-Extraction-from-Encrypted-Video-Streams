# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 16:25:03 2018


@author: Adam
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)


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



def ConfusionMatrixMet(cm):
    """
    FP = cm.sum(axis=0) - np.diag(confusion_matrix)  
    FN = cm.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(cm)
    TN = cm.values.sum() - (FP + FN + TP)
    
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    
    
    print('Accuracy = ' + ACC )
    """
    
    """   
    Recall =    (TP)/(TP + FN)
    Precision = (TP)/(TP + FP)
    
    F1 = (2*Precision*Recall)/(Precision + Recall)
    """
    REC = (np.diag(cm))/np.sum(cm, axis = 1)*100
    PRE = (np.diag(cm)/ np.sum(cm, axis = 0))*100
    
    """
    print('\nRecall = ')
    print(REC)
    print('\nPrecision = ')
    print(PRE)
    """
    
    F1 = (2*PRE*REC)/(PRE + REC)
    print('\nF1 = ')
    print(F1)
    print('\n')
    
    
    return F1





def PlotXYZ(x, y, z, ptype = 'markers', title = 'plot' , xn = 'Sample' , yn = 'Value' , ylim = 'y', name = 'temp-plot.html', zhist = False ):
        
        
    if(ylim == 'y'):
        ylim = [0 , np.max(y)]
    
    x = np.array(x,).astype(int)
    y = np.array(y).astype(int)
    z = np.array(z).astype(int)
    
    uz = np.unique(z)
    traces = list()
    tlab = uz[0]
    
    if(ptype == 'bar'):
        for tlab in uz:
            idx = np.where(z==tlab)[0]
            
            tx = np.array(x)[idx]
            ty = np.array(y)[idx]
            tz = tlab
            traces.append(go.Bar(x=tx, y=ty, name = tz)  )
    else:
                
        for tlab in uz:
            idx = np.where(z==tlab)[0]
            
            tx = np.array(x)[idx]
            ty = np.array(y)[idx]
            
            if(zhist == False):
                tz = tlab
            else:
                tz = str(tlab) + ' ' + str( int(((np.sum(z == tlab))/len(z))*100)/100 ) + '%' 
            traces.append(go.Scatter(x=tx, y=ty, mode=ptype, name = tz)  )
        
    data=go.Data(traces)
    layout=go.Layout(title=title, xaxis={'title':xn}, yaxis={'title':yn, 'range':ylim } )
    figure=go.Figure(data=data,layout=layout)
    plotly.offline.plot(figure, filename=name)

    
    
    
    
    
    
    return


def PlotXYZC(x, y, z, c, ptype = 'markers', title = 'plot' , xn = 'f1' , yn = 'f2', zn = 'Value' , name = 'temp-plot.html', chist = False, proba=[] ):
        
        
    
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    c = np.array(c)
    proba = np.array(proba)
    proba = np.round(proba,2)
    
    """
    idxs = np.argsort(c)
    x = x[idxs]
    y = y[idxs]
    z = z[idxs]
    c = c[idxs]
    print('sorted')
    """
    uc = np.unique(c)
    traces = []
    
    tlab = uc[0]
    for tlab in uc:
        print('tlab',tlab)
        idx = (c == tlab)
        
        tx = x[idx]
        ty = y[idx]
        tz = z[idx]
        
        if(chist == False):
            tc = tlab
        else:
            tc = str(tlab) + ' ' + str( int(((np.sum(c == tlab))/len(c))*100)) + '%' 
        
        if( len(proba) != 0):
            traces.append(go.Scatter3d(x=tx, y=ty, z=tz, mode=ptype, name = tc, text = proba[idx]))
        else:
            traces.append(go.Scatter3d(x=tx, y=ty, z=tz, mode=ptype, name = tc))
        
    data=go.Data(traces)
    layout=go.Layout(scene = dict(xaxis={'title':xn}, yaxis={'title':yn }, zaxis={'title':zn}), title=title )
    figure=go.Figure(data=data,layout=layout)
    plotly.offline.plot(figure, filename=name)




    
    
    return









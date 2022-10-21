## https://docs.google.com/document/d/1miSm10QFtcQGJby44PZGwgrME9q0FH2t6ZgArENuc2Q/edit
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from autoviz.AutoViz_Class import AutoViz_Class
import os
import numpy as np

#set and sort dataframe
df=pd.read_csv('DATA.csv',sep=',')
df["DATE"] = pd.to_datetime(df["DATE"])
#df['year']=pd.DatetimeIndex(df["DATE"]).year
df=df.sort_values(by=["DATE"],ascending=True)
# table=pd.pivot_table(df,values='UNRATE(%)',index=['year'],aggfunc=np.mean)
# df=table.reset_index()
# df.columns=['year','mean unemployment rate']
# print(df)

# df2=pd.read_csv('arrests_national_adults.csv',sep=',')
# #df2=df2.sort_values(by=["year"],ascending=True)
# print(df2)

# mergeddf=df2.merge(df,how='left',on=['year'])
# mergeddf=mergeddf.sort_values(by=["year"],ascending=True)
# mergeddf=

def mergethem():
    dflist=[]
    for i in ['corruption.csv','cost_of_living.csv','richest_countries.csv','tourism.csv','unemployment.csv']:
        dflist.append(pd.read_csv(i))

    mergeddf=dflist[0]
    for i in range(1,len(dflist)):
        mergeddf=mergeddf.merge(dflist[i],how='inner',on=['country'])

    mergeddf.to_csv('MERGED.csv')

## correlation matrix heatmap
def MatrixHeatmap():
    corr_matrix = df.corr()
    sn.heatmap(corr_matrix, annot=True)
    plt.show()

## Time plot of var unemployment
def UnemploymentPlot():
    plt.plot(df["DATE"],df['UNRATE(%)'])
    #plt.xticks(fontsize=5,rotation=90)
    plt.show()

#visualization
def AutoVizTest(ForceGen):
    if os.path.exists('./Visuals')==False or ForceGen==True:
        AV = AutoViz_Class()
        AV.AutoViz(filename='',dfte=df,verbose=1,chart_format='html',depVar='UNRATE(%)',save_plot_dir='./Visuals')
    else:
        print('Visuals already exist, and it takes a while to generate them. You can coerce this function by passing in True as the first positional argument')

#AutoVizTest(True)

def TimeEncoding(datelist):
    outputlist=[]
    for i in range(1,(len(datelist)+1)):
        outputlist.append(float(1/i))
    return outputlist

def StandardizeList(TargetList):
    outputlist=[]
    themean=np.mean(TargetList)
    thestd=np.std(TargetList)
    for i in TargetList:
        outputlist.append((i-themean)/thestd)
    return outputlist

def Partition(TargetList):
    TrainingSizePCT=0.75
    TestSizePCT=0.25
    SetSize=len(TargetList)
    TrainingSize=round(TrainingSizePCT*SetSize)
    TestSize=round(TestSizePCT*SetSize)
    OutputTrainData=[]
    OutputTestData=[]

    for i in range(0,TrainingSize):
        OutputTrainData.append(TargetList[i])
    
    for i in range(1,TestSize):
        idx=i+TrainingSize
        OutputTestData.append(TargetList[idx])
    
    return OutputTrainData,OutputTestData



def Neuralize():
    InputTrainData,OutputTrainData,InputTestData,OutputTestData=PrepareNeuralNetworkData(df,'QUATERLY REAL GDP')
    training_set_inputs = array(InputTrainData)
    training_set_outputs = array([OutputTrainData]).T
    random.seed(1)
    synaptic_weights = 2 * random.random((2, 1)) - 1
    for iteration in range(10000):
        output = 1 / (1 + exp(-(dot(training_set_inputs, synaptic_weights))))
        synaptic_weights += dot(training_set_inputs.T, (training_set_outputs - output) * output * (1 - output))
    print(1 / (1 + exp(-(dot(array([1, 0, 0]), synaptic_weights)))))

print(TimeEncoding(df['DATE'].tolist()))
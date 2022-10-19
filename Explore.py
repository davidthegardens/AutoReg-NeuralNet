## https://docs.google.com/document/d/1miSm10QFtcQGJby44PZGwgrME9q0FH2t6ZgArENuc2Q/edit
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from autoviz.AutoViz_Class import AutoViz_Class
import os
from numpy import exp, array, random, dot

#set and sort dataframe
df=pd.read_csv('DATA.csv',sep=',')
df["DATE"] = pd.to_datetime(df["DATE"])
df=df.sort_values(by=["DATE"],ascending=True)
print(df)

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

#AutoVizTest(False)

def PrepareNeuralNetworkData(df,TargetColumn):
    TrainingSizePCT=0.75
    TargetList=df[TargetColumn].to_list()
    DATEList=df['DATE'].to_list()
    OutputList=df['UNRATE(%)'].to_list()
    SetSize=len(TargetList)
    TestSizePCT=0.25
    TrainingSize=round(TrainingSizePCT*SetSize)
    TestSize=round(TestSizePCT*SetSize)
    InputTrainData=[]
    OutputTrainData=[]
    InputTestData=[]
    OutputTestData=[]

    for i in range(0,TrainingSize):
        InputTrainData.append([DATEList[i],TargetList[i]])
        OutputTrainData.append(OutputList[i])
    
    for i in range(1,TestSize):
        idx=i+TrainingSize
        InputTestData.append([DATEList[idx],TargetList[idx]])
        OutputTestData.append(OutputList[idx])
    
    return InputTrainData,OutputTrainData,InputTestData,OutputTestData


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
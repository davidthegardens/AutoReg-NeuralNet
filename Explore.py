## https://docs.google.com/document/d/1miSm10QFtcQGJby44PZGwgrME9q0FH2t6ZgArENuc2Q/edit
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from autoviz.AutoViz_Class import AutoViz_Class
import os
import numpy as np
from sklearn.preprocessing import normalize

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

## Must Cache Date0 for decoding
def TimeEncoding(datelist):
    outputlist=[]
    for i in range(1,(len(datelist)+1)):
        outputlist.append(float(1/i))
    return outputlist

## Must cache min max for decoding
def TimePartition(df):
    ##testsize is population-TrainingSize
    TrainingSizePCT=0.75
    SetSize=len(df)
    TrainingSize=int(round(TrainingSizePCT*SetSize))
    TrainingData=df.head(TrainingSize)
    TestData=df.iloc[TrainingSize:]
    return TrainingData,TestData

def Squishify(x):
    return 1/(1+np.exp(-x))

def Encode(df):
    colnames=list(df.columns.values)
    for i in colnames:
        if df[i].dtypes in ['float64','int64']:
            ###CHANGE RESCALING TO [-5,5] FROM IMPROVED ACCURACY POST-SIGMOID
            #### potential problem if test data surpasses min/max? 
            #Min max feature rescaling between -5 and 5
            df['new '+i]=-5+(((df[i]-np.min(df[i]))*-10)/(np.max(df[i])-np.min(df[i])))
            #sigmoid
            df["new "+i]=Squishify(df['new '+i])
    return df


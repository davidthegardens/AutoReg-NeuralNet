## https://docs.google.com/document/d/1miSm10QFtcQGJby44PZGwgrME9q0FH2t6ZgArENuc2Q/edit
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
# from autoviz.AutoViz_Class import AutoViz_Class
import os
import numpy as np
from sklearn.preprocessing import normalize
import math
from sklearn.linear_model import LinearRegression
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

def ReplaceZero(x,base):
    if x==0:
        return np.nan
    else: 
        return x**base

#set and sort dataframe
df=pd.read_csv('CovidData.csv',sep=',')
#df["DATE"] = pd.to_datetime(df["DATE"])
#df['year']=pd.DatetimeIndex(df["DATE"]).year
df=df.sort_values(by=["Date Index"],ascending=True)
cleancopy=df
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
# def AutoVizTest(ForceGen):
#     if os.path.exists('./Visuals')==False or ForceGen==True:
#         AV = AutoViz_Class()
#         AV.AutoViz(filename='',dfte=df,verbose=1,chart_format='html',depVar='UNRATE(%)',save_plot_dir='./Visuals')
#     else:
#         print('Visuals already exist, and it takes a while to generate them. You can coerce this function by passing in True as the first positional argument')

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

def DateIndexer(df,xcolname):
    DATEList=df[xcolname].tolist()
    DATEIdx=[]
    for i in DATEList:
        DATEIdx.append(abs((i - DATEList[0]).days))
    df[xcolname]=DATEIdx
    return df

def TestBases(decimals,xrange,operation):
    Relist=[]
    xrange.sort()
    Size=abs(xrange[1]-xrange[0])
    Steps=int(Size/decimals)
    for x in range(0,Steps+1):
        Relist.append(float(round((max(xrange)-decimals*x),int(math.log(1/decimals,10)))))
    if operation=='log':
        if min(Relist)<0:
            raise Exception("Cannot compute negative log bases, change base range.")
        if 0 in Relist:
            raise Exception("Log base 0 evaluates to undefined, adjust base range.")
        if 1 in Relist:
            raise Exception("Log base 1 evaluates to negative infinity, and is undefined. Consider adjusting base range.")
    if operation=='lag':
        if int(decimals)!=decimals:
            raise Exception("The lag operation only accepts integers.")

    return Relist

#print(TestBases(0.01,[-1,1]))

def TestLinearity(testdf,xcolname):
    lm=LinearRegression()
    lm.fit(testdf[[xcolname]],testdf['Transformed'])
    coeff=lm.coef_[0]
    intercept=lm.intercept_
    testdf['FORECAST']=testdf[xcolname]*coeff+intercept
    testdf['MAPE']=abs(testdf['Transformed']-testdf['FORECAST'])/abs(testdf['Transformed'])
    testdf['SE']=(testdf['Transformed']-testdf['FORECAST'])**2
    testdf['ST']=(testdf['Transformed']-np.mean(testdf['Transformed']))**2

    return np.mean(testdf['MAPE']),len(testdf),1-(np.sum(testdf['SE'])/np.sum(testdf['ST']))

def Logger(x,base):
    if x==0:
        ####This should always evaluate to -infinity but this would destroy any chance at evaluating models
        return np.nan
    else:
        return math.log(x,base)

def Shifter(testdf,ycolname,base,potenshift,operation):
    if operation == 'log':
        if potenshift<0:
            print('shifted')
            testdf['Transformed']=testdf.apply(lambda row: Logger((row[ycolname]-potenshift),base),axis=1)
            testdf['Transformed']=testdf['Transformed']+potenshift
        else:
            testdf['Transformed']=testdf.apply(lambda row: Logger(row[ycolname],base),axis=1)
            print('did it work')
    elif operation == 'power':
        testdf['Transformed']=testdf.apply(lambda row: ReplaceZero(row[ycolname],base),axis=1)
        ##somehow this (below) is okay. Sqrt(-1)=Undefined but -1**0.5=-1. I tried using this to cause a math error but instead it just worked, so here we are. I'm not about to code to turn float into fraction, then check odd or even of it's denominator, so this is staying.
    #print(testdf)
    elif operation == 'lag':
        testdf['Transformed']=testdf[ycolname].shift(int(base))
    testdf.dropna(inplace=True)
    #testdf.fillna(testdf.mean(), inplace=True)
    return testdf

def EvaluateBases(decimals,rangex,df,xcolname,ycolname,operation):
    if df[xcolname].dtypes=='datetime64[ns]':
        df=DateIndexer(df,xcolname)
    TestBaseList=TestBases(decimals,rangex,operation)
    potenshift=np.min(df[ycolname])
    dftemplate=pd.DataFrame({ycolname:df[ycolname],xcolname:df[xcolname]})
    MAPEList=[]
    nList=[]
    r2List=[]
    for base in TestBaseList:
        if base==0 and operation=='power':
            Mape=0
            n=0
            r2=0
        else:
            dftemplate=pd.DataFrame({ycolname:df[ycolname],xcolname:df[xcolname]})
            testdf=Shifter(dftemplate,ycolname,base,potenshift,operation)
            Mape,n,r2=TestLinearity(testdf,xcolname)
        MAPEList.append(Mape)
        nList.append(n)
        r2List.append(r2)
        # plt.scatter(testdf[xcolname],testdf['Transformed'])
       # plt.scatter(testdf[xcolname],testdf['Transformed'])
        # plt.scatter(testdf[xcolname],testdf['unlogged forecast'])
        #plt.scatter(testdf[xcolname],testdf['FORECAST'])
    if operation=='power':
        dftemplate=pd.DataFrame({ycolname:df[ycolname],xcolname:df[xcolname]})
        testdf=Shifter(dftemplate,ycolname,3,potenshift,'log')
        Mape,n,r2=TestLinearity(testdf,xcolname)
        MAPEList.append(Mape)
        nList.append(n)
        r2List.append(r2)
        TestBaseList.append('Log(x)')

    Results=pd.DataFrame({operation:TestBaseList,'MAPE':MAPEList,'n':nList,'R2':r2List,'Transformed Variable':ycolname})
    #Results.to_csv(str(operation)+str(ycolname)+str(xcolname))
    #plt.savefig('scat.png')
    BestRow=Results[Results['R2']==np.max(Results['R2'])]
    BestTrans=BestRow[operation].tolist()[0]
    if operation=='power':
        Include=Results[Results['R2']==np.max(Results['R2'])]
        BestPower=Include['power'].tolist()[0]
        if BestPower!="Log(x)":
            if BestPower>1:
                InclusionList=list(range(1,math.trunc(BestPower)+1))
                InclusionList.append(BestPower)
                BestR2=Include['R2'].tolist()[3]
                BestOp=Include[operation].tolist()[0]
        else:
            BestR2=None
            BestOp=None
            InclusionList=None
    else:
        BestR2=None
        BestOp=None
        InclusionList=None
    
    #return Results,BestR2,BestOp,InclusionList
    return BestTrans

#decided transformations

columns=list(df.columns)
columns.remove('DailyDeaths')
translist=[]
for col in columns:
    print('Lagging '+col)
    Results=EvaluateBases(1,[-30,30],df,'DailyDeaths',col,'lag')

    if Results not in [-30,30]:
        translist.append(Results)
        cleancopy[col].shift(int(Results))
    else:
        translist.append('Requires Further Testing')

    print('Completed')
cleancopy.to_csv("LaggedDataset.csv")
Transformations=pd.DataFrame({'Var':columns,'Lag':translist})

translist=[]
for col in columns:
    print('Powering '+col)
    df=pd.read_csv("LaggedDataset.csv")
    Results=EvaluateBases(0.1,[-4,4],df,'DailyDeaths',col,'power')
    translist.append(Results)
    if Results=="Log(X)":
        tempdf=Shifter(df,col,3,np.min(df[col]),'log')
        df[col]=tempdf['Transformed']
    elif Results not in [-4,4]:
        df[col]=df.apply(lambda row: ReplaceZero(row[col],float(Results)),axis=1)
    print('Completed')
df.to_csv("PoweredLaggedDataset.csv")
Transformations['Power']=translist
Transformations.to_csv('Transformations',sep=',')
# print(EvaluateBases(0.1,[-4,4],df,'DailyDeaths','DailyActive','power'))

def OptimizationScan(df,xcolname,ycolname):
    r2s=[]
    r2snames=['lag','power','power','log']
    BestChanges=[]
    Results,r2,BestChange=EvaluateBases(1,[-100,100],df,xcolname,ycolname,'lag')
    r2s.append(r2)
    BestChange.append(BestChange)
    Results,rootr2,BestChange=EvaluateBases(0.1,[-1,1],df,xcolname,ycolname,'power')
    r2s.append(rootr2)
    BestChange.append(BestChange)
    Results,expr2,BestChange=EvaluateBases(1,[-20,20],df,xcolname,ycolname,'power')
    r2s.append(expr2)
    BestChange.append(BestChange)
    Results,logr2,BestChange=EvaluateBases(1,[2,2],df,xcolname,ycolname,'log')
    tempdf=cleancopy[xcolname,ycolname]
    a,b,regr2=TestLinearity(tempdf,xcolname)
    BestChange.append(BestChange)
    if logr2>regr2:
        r2s.append(logr2)
    else:
        r2s.append(0)
    Choices=pd.DataFrame({'Operation':r2snames,'R2':r2s,'Best Change':BestChanges})
    Choice=Choices[Choices['R2']==np.max(Choices['R2'])]



#print(EvaluateBases(1,[-100,100],df,"DATE","CONSUMER CONF INDEX",'lag'))


#Operations: power,log,lag
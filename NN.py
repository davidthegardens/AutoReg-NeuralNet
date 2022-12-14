# Import required libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
import random

# Import necessary modules
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from NNAR import netwrapper,predict_from_load



def Role(x,TrainSize):
    if x<=TrainSize:
        val="TRAIN"
    else: 
        val="TEST"
    return val

#Regular sigmoid but scaled [-1,1]
def Squishify(x):
    return (2/(1+np.exp(-x)))-1

#Inverse scaled sigmoid // scaled logit
def Unsquish(x):
    return np.log(1/((2/(x+1))-1))

#Feature Scaler
def MinMaxScaler(df,colname,min,max):
    currentmin=np.min(df[colname])
    currentmax=np.max(df[colname])
    df[colname]=min+(((df[colname]-currentmin)*(max-min))/(currentmax-currentmin))
    return df,currentmin,currentmax

def DatetoNumeric(df,colname,coerce):
    if coerce ==True:
        df[colname]=pd.to_datetime(df[colname])
    DateZero=np.min(df[colname])
    df[colname]=(df[colname]-DateZero).dt.days
    return df

def DataCheck(df):
    errorlist=[]
    tempdf=df
    for col in tempdf:
        datatype=df[col].dtypes
        if  datatype not in ['int64','float64','datetime64[ns]']:
            errorlist.append(col)
        elif datatype=='datetime64[ns]':
            df=DatetoNumeric(df,col,False)        
    tempdf=None

    if len(errorlist)!=0:
        print(errorlist)
        resp=str(input('The above variable(s) are not acceptable datatype(s). Enter d to remove them, of any key to exit. '))
        if resp != 'd':
            raise Exception("Time to fix your data.")
        else:
            df=df.drop(columns=errorlist)
    errorlist=None
    return df

def Encoder(df,squish):
    df=DataCheck(df)
    tempdf=df
    DecodeTable=[[],[],[]]
    for colname in tempdf:
        #purely objectively, I like range [-5,5]
        df,currentmin,currentmax=MinMaxScaler(df,colname,-1,1)
        DecodeTable[0].append(colname)
        DecodeTable[1].append(currentmin)
        DecodeTable[2].append(currentmax)
        if squish==True:
            df[colname]=Squishify(df[colname])
    tempdf=None
    return df,DecodeTable

def Decoder(df,DecodeTable,ycolname,squish):
    for idx in range(len(DecodeTable[0])):
        colname=DecodeTable[0][idx]
        min=DecodeTable[1][idx]
        max=DecodeTable[2][idx]
        if squish==True:
            df[colname]=Unsquish(df[colname])
        df,currentmin,currentmax=MinMaxScaler(df,colname,min,max)
        if ycolname != None and ycolname==colname:
            if squish==True:
               df['Prediction']=Unsquish(df['Prediction'])
            df,currentmin,currentmax=MinMaxScaler(df,'Prediction',min,max)

    return df

def PrepareData(Trainpct,df,ycolname,forcedsize):
    IndependentVariables=list(df.columns)
    IndependentVariables.remove(ycolname)
    TrainSize=int(np.trunc(len(df)*Trainpct))
    if forcedsize!=None:
        TrainSize=forcedsize
    TestSize=len(df)-TrainSize
    dfTrain=df.head(TrainSize)
    dfTest=df.tail(TestSize)
    X_train=dfTrain[IndependentVariables].values
    X_test=dfTest[IndependentVariables].values
    y_train=dfTrain[ycolname].values
    y_test=dfTest[ycolname].values
    return X_train, X_test, y_train, y_test

def GetStructure():
    Layers=random.randrange(2,10)
    paramx=[]
    paramstr=[]
    for i in range(Layers+1):
        rand=random.randrange(3,50)
        paramx.append(rand)
        paramstr.append(str(rand))
    structure=paramx
    namecomp="_".join(paramstr)
    print(structure)
    return structure,namecomp

def fix(fixthis):
    lister=[]
    for i in range(len(fixthis)):
        lister.append([fixthis[i]])
    return np.array(lister,dtype=float)


df = pd.read_csv('C:\\477\\Team Project\\bixidata\\SeasonalityRegression.csv')
#df=df[['Date','Year','Month','Day of Week','Hour','Count of Trips']]
#df=df.drop(columns=['Timestamp'])
#df=DatetoNumeric(df,'Date',True)
#df=df[['Date','Year','Month','Day of Week','Hour','Count of Trips']]
# df=df.head(100)

# df['Lagged y k=1']=df['Count of Trips'].shift(1)
# df['Lagged y k=1']=df['Lagged y k=1'].fillna(0)
# df['Lagged y k=2']=df['Count of Trips'].shift(2)
# df['Lagged y k=2']=df['Lagged y k=2'].fillna(0)
#df['index'] = range(1, len(df) + 1)
df,DecodeTable=Encoder(df,False)
X_train, X_test, y_train, y_test=PrepareData(0.8,df,'Count of Trips',14377)

def predictfromload(df,filename):
    hash='aaa'
    out1,namecomp=predict_from_load(fix(X_train),True,filename,1)
    out2,namecomp=predict_from_load(fix(X_test),True,filename,1)
    out=np.append(out1,out2)
    df['Prediction']=np.reshape(out,out.shape[0])
    df=Decoder(df,DecodeTable,'Count of Trips',False)
    df['SE']=(df['Count of Trips']-df['Prediction'])**2
    df['APE']=abs(df['Count of Trips']-df['Prediction'])/df['Count of Trips']
    df['Prediction'].replace([np.inf, -np.inf],np.nan, inplace=True)
    mape=round(np.nanmean(df['APE'].tail(5160))*100)
    mse=round(np.nanmean(df['SE'].tail(5160))*100)
    print('MAPE=',mape,'MSE=',mse)
    df.to_csv('C:\\477\\Team Project\\bixidata\\NNoutput\\NNAROutput_'+namecomp+'_'+hash+'_'+str(mape)+'.csv')

#predictfromload(df,'C:/477/Team Project/bixidata/NNoutput/NNARparameters0952d5b640d2433ca7074ebadd837e3d.pkl')
####best so far C:/477/Team Project/bixidata/NNoutput/NNARparameters5edb2084be3b4681bdf5a75ac3b087b6.pkl
def NNAR(df):
    recordmape=69
    for i in range(1000):
        structure,namecomp=GetStructure()
        out1,out2,hash=netwrapper(fix(X_train),fix(fix(y_train)),fix(X_test),structure,autoregress=True,epochs=1000,learning_rate=0.1,dynamic_learning=False,verbose=True,early_modelling=False,location="C:/477/Team Project/bixidata/NNoutput/NNARparameters",ImprovementThreshold=0.1)
        if out1=='Nothing':
            continue
        out=np.append(out1,out2)
        #print(out.shape)
        df['Prediction']=np.reshape(out,out.shape[0])
        df=Decoder(df,DecodeTable,'Count of Trips',False)
        
        try:
            df['SE']=(df['Count of Trips']-df['Prediction'])**2
            df['APE']=abs(df['Count of Trips']-df['Prediction'])/df['Count of Trips']
            mape=round(np.nanmean(df['APE'].tail(5160))*100)
            mse=round(np.nanmean(df['SE'].tail(5160))*100)
        except Exception:
            mape=420
            mse=420
        
        print(mape,mse)
        if mape<recordmape:
            recordmape=mape
            df.to_csv('C:\\477\\Team Project\\bixidata\\NNoutput\\NNAROutput_'+namecomp+'_'+hash+'_'+str(mape)+'.csv')

NNAR(df)
# mapelist=[0.8]
# for i in range(1000):
#     structure,namecomp=GetStructure()
#     # namecomp=[8,32,28,59,82,22,87,86,29]
#     # structure=(8,32,28,59,82,22,87,86,29)
#     mlp = MLPRegressor(hidden_layer_sizes=structure, activation='relu', solver='adam',shuffle=True, max_iter=500000,learning_rate='invscaling',early_stopping=True,)
#     mlp.fit(X_train,y_train)
    

#     predict_train = mlp.predict(X_train)
#     predict_test = mlp.predict(X_test)

#     fullprediction=np.append(predict_train,predict_test)
#     df['Prediction']=fullprediction

#     # df=Decoder(df,DecodeTable,'Count of Trips',False)
#     df['APE']=abs(df['Count of Trips']-df['Prediction'])/df['Count of Trips']
#     # df['Role']=Role(df['index'],14377)
#     Mape=np.mean(df['APE'].tail(len(df)-14377))
#     print(Mape)

#     if Mape<min(mapelist):
#         mapelist.append(Mape)
#         modeloutp=pd.DataFrame({'Weight':mlp._best_coefs,'Bias':mlp._best_intercepts})
#         print(df)
#         filename=namecomp+'_'+str(round(Mape*100))+'.csv'
#         modeloutp.to_csv('C:\\477\\Team Project\\bixidata\\NNoutput\\NolagNNweights'+filename)
#         df.to_csv('C:\\477\\Team Project\\bixidata\\NNoutput\\NolagNN'+filename)
#     else: print('bad model')
        
# print(Decoder(df,DecodeTable,'Count of Trips'))
# print(confusion_matrix(y_train,predict_train))
# print(classification_report(y_train,predict_train))
# df=DatetoNumeric(df,'Date',True)
# Encoded,DecodeTable=Encoder(df)
# print(Encoded)
# df=Decoder(Encoded,DecodeTable)
# print(df)



# print(df.shape)
# dfdata=df.describe().transpose()
# dfdata['Upper']=dfdata['mean']+2*dfdata['std']
# dfdata['Lower']=dfdata['mean']-2*dfdata['std']


import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta

Sheets=["C:\\477\\Team Project\\bixidata\\OD_2014.csv","C:\\477\\Team Project\\bixidata\\OD_2015.csv","C:\\477\\Team Project\\bixidata\\OD_2016.csv","C:\\477\\Team Project\\bixidata\\OD_2017.csv"]

def ProcessSheet(path,fullyear):

    df=pd.read_csv(path)
    df['start_date']=pd.to_datetime(df['start_date'])

    if fullyear==True and path != Sheets[len(Sheets)-1]:
        EndDate=np.min(df['start_date'])+ relativedelta(years = 1)
        EndDate=EndDate-datetime.timedelta(days=1)
    else:
        EndDate=np.max(df['start_date'])

    expected_df=pd.DataFrame({'Timestamp':pd.date_range(np.min(df['start_date']),EndDate,freq='H')})
    expected_df['Date']=expected_df['Timestamp'].dt.date
    expected_df['hour']=expected_df['Timestamp'].dt.hour
    df['Date']=df['start_date'].dt.date
    df['hour']=df['start_date'].dt.hour
    df=df.groupby(['Date','hour']).agg(np.count_nonzero)
    final_df=pd.merge(df,expected_df,how='right',on=['Date','hour'])
    final_df=final_df[['Timestamp','Date','hour','start_date']]
    final_df['Year']=final_df['Timestamp'].dt.year
    final_df['Month']=final_df['Timestamp'].dt.month
    final_df['Day of Year']=final_df['Timestamp'].dt.day_of_year
    final_df['Day of Week']=final_df['Timestamp'].dt.day_of_week
    final_df['Hour']=final_df['Timestamp'].dt.hour
    final_df['Count of Trips']=final_df['start_date']
    final_df.drop(columns=['start_date','hour'],inplace=True)
    final_df.fillna(0,inplace=True)
    return final_df

def ConcatenateSets(Sheets,fullyear):
    FinalDF=ProcessSheet(Sheets[0],fullyear)
    for idx in range(1,len(Sheets)):
        TempDF=ProcessSheet(Sheets[idx],fullyear)
        FinalDF=pd.concat([FinalDF,TempDF])
    FinalDF.to_csv("C:\\477\\Team Project\\bixidata\\BixiDataFull.csv")

ConcatenateSets(Sheets,True)


## https://docs.google.com/document/d/1miSm10QFtcQGJby44PZGwgrME9q0FH2t6ZgArENuc2Q/edit
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

df=pd.read_csv('DATA.csv',sep=',')
df["DATE"] = pd.to_datetime(df["DATE"])
df=df.sort_values(by=["DATE"],ascending=True)
print(df)

## correlation matrix heatmap
def MatrixHeatmap():
    corr_matrix = df.corr()
    sn.heatmap(corr_matrix, annot=True)
    plt.show()


plt.plot(df["DATE"],df['UNRATE(%)'])
#plt.xticks(fontsize=5,rotation=90)
plt.show()
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.dates import HourLocator, MinuteLocator, DateFormatter, drange

filename1 = 'labels5-5.csv'
filename2 = 'pred5-5.csv'
filename3 = 'pred2-5.csv'
df1 = pd.read_csv(filename1)
df2 = pd.read_csv(filename2)
df3 = pd.read_csv(filename3)

def printFiguresOfFre(df1, df2, df3, length):
    df1 = df1.iloc[:, 1:2]
    df2 = df2.iloc[:, 1:2]
    df3 = df3.iloc[:, 1:2]
    # df4 = df4.iloc[:, 1:2]
    # df5 = df5.iloc[:, 1:2]

    i = 0
    xxaxis = []
    yxaxis = []
    yyaxis = []
    yzaxis = []
    # yaaxis = []
    # ybaxis = []
    df = []
    for j in range(length):
        xxaxis.append(j+1)
        yxaxis.append(df1.iloc[j, 0])
        yyaxis.append(df2.iloc[j, 0])
        yzaxis.append(df3.iloc[j, 0])
        # yaaxis.append(df4.iloc[j + 5, 0])
        # ybaxis.append(df5.iloc[j + 11, 0])
    xxaxis = np.array(xxaxis).reshape(len(xxaxis), 1)
    yxaxis = np.array(yxaxis).reshape(len(yxaxis), 1)
    yyaxis = np.array(yyaxis).reshape(len(yyaxis), 1)
    # yaaxis = np.array(yaaxis).reshape(len(yaaxis), 1)
    # ybaxis = np.array(ybaxis).reshape(len(ybaxis), 1)
    # df = np.hstack((xxaxis, yxaxis, yyaxis, yzaxis))
    # df = pd.DataFrame(df)
    # df.to_csv('csvs/results--'+str(i+1)+'.csv')
    plt.plot(xxaxis,yxaxis,color='black', linewidth = 0.7, marker='')
    plt.plot(xxaxis,yyaxis,color='grey', linewidth = 0.7, marker='')
    # plt.plot(xxaxis,yaaxis,color='yellow', linewidth = 0.5, marker='')
    # plt.plot(xxaxis,ybaxis,color='grey', linewidth = 0.5, marker='')
    plt.legend(['Real Value', 'Graph WaveNet'])
    plt.xlabel('Time')
    plt.ylabel('Value(KPa)')
    plt.savefig('figures/preds vs. real'+ filename2 +'.png')
    plt.show()
    plt.close()

if __name__ == "__main__":
    printFiguresOfFre(df1, df2, df3, 288)
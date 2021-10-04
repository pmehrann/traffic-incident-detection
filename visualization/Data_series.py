import numpy as np
import pandas as pd

# load data
num_features = 3
data_6368 = pd.read_csv('E:/MSc/Crash Detection/Data/Site1/East/6375.csv')
time = data_6368.iloc[2:,1]
date = data_6368.iloc[1,2:-1]
data_6368 = data_6368.iloc[2:,2:-1]  # [2:, 1:] for labels and [2:, 2:-1] for sensors data
data_6368 = data_6368.astype('float32')
len_day = data_6368.shape[0]    # number of acquired data during one day
num_days = data_6368.shape[1]//num_features    # number of days

# Broadcast data of days into consecutive intervals
data = pd.DataFrame({"Flow":[], "Occupancy":[], "Speed":[]})
columns = np.array(["Flow", "Occupancy", "Speed"])
for i in range(num_features):

    data_years = data_6368.iloc[:,i*num_days:(i+1)*num_days]
    data_years = np.array(data_years)
    temp = data_years[:,0]

    for j in range(1,num_days):
        temp = np.append(temp, data_years[:,j])

    data[columns[i]] = temp

# save data series to csv file
data.to_csv("E:/MSc/Crash Detection/Data/Site1/East/6375_series.csv")





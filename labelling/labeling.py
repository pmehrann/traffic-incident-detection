import numpy as np
import pandas as pd

# load data
data_6368 = pd.read_csv('E:/MSc/Crash Detection/Data/Site1/East/6368_series.csv')
data_6369 = pd.read_csv('E:/MSc/Crash Detection/Data/Site1/East/6369_series.csv')
data_6370 = pd.read_csv('E:/MSc/Crash Detection/Data/Site1/East/6370_series.csv')
data_6371 = pd.read_csv('E:/MSc/Crash Detection/Data/Site1/East/6371_series.csv')
data_6372 = pd.read_csv('E:/MSc/Crash Detection/Data/Site1/East/6372_series.csv')
data_6373 = pd.read_csv('E:/MSc/Crash Detection/Data/Site1/East/6373_series.csv')
data_6374 = pd.read_csv('E:/MSc/Crash Detection/Data/Site1/East/6374_series.csv')
data_6375 = pd.read_csv('E:/MSc/Crash Detection/Data/Site1/East/6375_series.csv')
labels = pd.read_csv('E:/MSc/Crash Detection/Data/Site1/East/labels_series.csv')

# Concatenate all sensors data for fusion
dataset = pd.concat([data_6368, data_6369, data_6370, data_6371, data_6372, data_6373, data_6374, data_6375, labels],
                    keys=['6368', '6369', '6370', '6371', '6372', '6373', '6374', '6375', 'labels'], axis=1)

# drop previous dataframes indexes
dataset.drop(columns=[('6368', 'Unnamed: 0'), ('6369', 'Unnamed: 0'), ('6370', 'Unnamed: 0'),
                      ('6371', 'Unnamed: 0'), ('6372', 'Unnamed: 0'), ('6373', 'Unnamed: 0'),
                      ('6374', 'Unnamed: 0'), ('6375', 'Unnamed: 0'), ('labels', 'Unnamed: 0')], inplace=True)

dataset.to_csv("E:/MSc/Crash Detection/Data/Site1/East/dataset.csv")
print(dataset.keys())




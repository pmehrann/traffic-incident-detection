import numpy as np
from numpy import cov
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load data
dataset = pd.read_csv('E:/MSc/Crash Detection/Data/Site1/East/dataset.csv', index_col=[0], header=[0,1])

# index of accidents
labels = dataset['labels']
temp = labels == 1
accidents = temp.index[temp['Labels']].tolist()
print(accidents)
print('number of accidents:', len(accidents))

# extract volume of each sensor
volume_6368 = dataset[('6368', 'Flow')]
volume_6369 = dataset[('6369', 'Flow')]
volume_6370 = dataset[('6370', 'Flow')]
volume_6371 = dataset[('6371', 'Flow')]
volume_6372 = dataset[('6372', 'Flow')]
volume_6373 = dataset[('6373', 'Flow')]
volume_6374 = dataset[('6374', 'Flow')]
volume_6375 = dataset[('6375', 'Flow')]

# extract occupancy of each sensor
Occupancy_6368 = dataset[('6368', 'Occupancy')]
Occupancy_6369 = dataset[('6369', 'Occupancy')]
Occupancy_6370 = dataset[('6370', 'Occupancy')]
Occupancy_6371 = dataset[('6371', 'Occupancy')]
Occupancy_6372 = dataset[('6372', 'Occupancy')]
Occupancy_6373 = dataset[('6373', 'Occupancy')]
Occupancy_6374 = dataset[('6374', 'Occupancy')]
Occupancy_6375 = dataset[('6375', 'Occupancy')]

# extract speed of each sensor
speed_6368 = dataset[('6368', 'Speed')]
speed_6369 = dataset[('6369', 'Speed')]
speed_6370 = dataset[('6370', 'Speed')]
speed_6371 = dataset[('6371', 'Speed')]
speed_6372 = dataset[('6372', 'Speed')]
speed_6373 = dataset[('6373', 'Speed')]
speed_6374 = dataset[('6374', 'Speed')]
speed_6375 = dataset[('6375', 'Speed')]

# extract data around the crash
l = 6
index = accidents[35]
vol_1 = volume_6368.iloc[index-l:index+l]
vol_2 = volume_6369.iloc[index-l:index+l]
vol_3 = volume_6370.iloc[index-l:index+l]
vol_4 = volume_6371.iloc[index-l:index+l]
vol_5 = volume_6372.iloc[index-l:index+l]
vol_6 = volume_6373.iloc[index-l:index+l]
vol_7 = volume_6374.iloc[index-l:index+l]
vol_8 = volume_6375.iloc[index-l:index+l]

occ_1 = Occupancy_6368.iloc[index-l:index+l]
occ_2 = Occupancy_6369.iloc[index-l:index+l]
occ_3 = Occupancy_6370.iloc[index-l:index+l]
occ_4 = Occupancy_6371.iloc[index-l:index+l]
occ_5 = Occupancy_6372.iloc[index-l:index+l]
occ_6 = Occupancy_6373.iloc[index-l:index+l]
occ_7 = Occupancy_6374.iloc[index-l:index+l]
occ_8 = Occupancy_6375.iloc[index-l:index+l]

sp_1 = speed_6368.iloc[index-l:index+l]
sp_2 = speed_6369.iloc[index-l:index+l]
sp_3 = speed_6370.iloc[index-l:index+l]
sp_4 = speed_6371.iloc[index-l:index+l]
sp_5 = speed_6372.iloc[index-l:index+l]
sp_6 = speed_6373.iloc[index-l:index+l]
sp_7 = speed_6374.iloc[index-l:index+l]
sp_8 = speed_6375.iloc[index-l:index+l]

# data visualization and correlation
plt.figure(1)
plt.plot(range(index-l,index+l),vol_1,marker='o')
plt.plot(range(index-l,index+l),vol_2,marker='^')
plt.plot(range(index-l,index+l),vol_3,marker='+')
plt.plot(range(index-l,index+l),vol_4,marker='*')
plt.plot(range(index-l,index+l),vol_5,marker='s')
plt.plot(range(index-l,index+l),vol_6,marker='p')
plt.plot(range(index-l,index+l),vol_7,marker='h')
plt.plot(range(index-l,index+l),vol_8,marker='d')
plt.xlabel('time intervals(5 min)')
plt.ylabel('Volume')
plt.legend(['6368', '6369', '6370', '6371', '6372', '6373', '6374', '6375'])
plt.show()

plt.figure(2)
plt.plot(range(index-l,index+l),occ_1,marker='o')
plt.plot(range(index-l,index+l),occ_2,marker='^')
plt.plot(range(index-l,index+l),occ_3,marker='+')
plt.plot(range(index-l,index+l),occ_4,marker='*')
plt.plot(range(index-l,index+l),occ_5,marker='s')
plt.plot(range(index-l,index+l),occ_6,marker='p')
plt.plot(range(index-l,index+l),occ_7,marker='h')
plt.plot(range(index-l,index+l),occ_8,marker='d')
plt.xlabel('time intervals(5 min)')
plt.ylabel('Occupancy')
plt.legend(['6368', '6369', '6370', '6371', '6372', '6373', '6374', '6375'])
plt.show()

plt.figure(3)
plt.plot(range(index-l,index+l),sp_1,marker='o')
plt.plot(range(index-l,index+l),sp_2,marker='^')
plt.plot(range(index-l,index+l),sp_3,marker='+')
plt.plot(range(index-l,index+l),sp_4,marker='*')
plt.plot(range(index-l,index+l),sp_5,marker='s')
plt.plot(range(index-l,index+l),sp_6,marker='p')
plt.plot(range(index-l,index+l),sp_7,marker='h')
plt.plot(range(index-l,index+l),sp_8,marker='d')
plt.xlabel('time intervals(5 min)')
plt.ylabel('Speed')
plt.legend(['6368', '6369', '6370', '6371', '6372', '6373', '6374', '6375'])
plt.show()

plt.figure(4)
sns.set(color_codes=True)
sns.distplot(vol_1)
plt.show()
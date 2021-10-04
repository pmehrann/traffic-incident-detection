import os
import pandas as pd
import numpy as np

# entries = os.listdir('E:/MSc/Crash Detection/Data/Site1/Mix/')

# entries = os.listdir('C:/Users/pouya/Desktop/Incident detection extension project/code python/20200325/data-20200317/')
main_dir='C:/Users/pouya/OneDrive - University of Waterloo/0-Projects/Incident detection extension project/code python/Labeling/files/'
entries = os.listdir(main_dir)
arrays = [['6368', '6368', '6368', '6368', '6368', '6368',
           '6369', '6369', '6369', '6369', '6369', '6369',
           '6370', '6370', '6370', '6370', '6370', '6370',
           '6371', '6371', '6371', '6371', '6371', '6371',
           '6372', '6372', '6372', '6372', '6372', '6372',
           '6373', '6373', '6373', '6373', '6373', '6373',
           '6374', '6374', '6374', '6374', '6374', '6374',
           '6375', '6375', '6375', '6375', '6375', '6375',
           '6426', '6426', '6426', '6426', '6426', '6426',
           '6427', '6427', '6427', '6427', '6427', '6427',
           '6428', '6428', '6428', '6428', '6428', '6428',
           '6429', '6429', '6429', '6429', '6429', '6429',
           '6430', '6430', '6430', '6430', '6430', '6430',
           '6431', '6431', '6431', '6431', '6431', '6431',
           '6432', '6432', '6432', '6432', '6432', '6432',
           '6433', '6433', '6433', '6433', '6433', '6433'],
         ['Capacity', 'Density', 'Flow', 'Headway', 'Occupancy', 'Speed',
          'Capacity', 'Density', 'Flow', 'Headway', 'Occupancy', 'Speed',
          'Capacity', 'Density', 'Flow', 'Headway', 'Occupancy', 'Speed',
          'Capacity', 'Density', 'Flow', 'Headway', 'Occupancy', 'Speed',
          'Capacity', 'Density', 'Flow', 'Headway', 'Occupancy', 'Speed',
          'Capacity', 'Density', 'Flow', 'Headway', 'Occupancy', 'Speed',
          'Capacity', 'Density', 'Flow', 'Headway', 'Occupancy', 'Speed',
          'Capacity', 'Density', 'Flow', 'Headway', 'Occupancy', 'Speed',
          'Capacity', 'Density', 'Flow', 'Headway', 'Occupancy', 'Speed',
          'Capacity', 'Density', 'Flow', 'Headway', 'Occupancy', 'Speed',
          'Capacity', 'Density', 'Flow', 'Headway', 'Occupancy', 'Speed',
          'Capacity', 'Density', 'Flow', 'Headway', 'Occupancy', 'Speed',
          'Capacity', 'Density', 'Flow', 'Headway', 'Occupancy', 'Speed',
          'Capacity', 'Density', 'Flow', 'Headway', 'Occupancy', 'Speed',
          'Capacity', 'Density', 'Flow', 'Headway', 'Occupancy', 'Speed',
          'Capacity', 'Density', 'Flow', 'Headway', 'Occupancy', 'Speed']]
tuples = list(zip(*arrays))
index1 = pd.MultiIndex.from_tuples(tuples)
features = pd.DataFrame(columns=index1)

for i, entry in enumerate(entries):

    if entry == 'Labels.csv':
        break

    print(entry)
    dir = main_dir + entry
    data = pd.read_csv(dir)
    data = data.iloc[2:, 2:-1]
    values = data[:].values
    temp = pd.DataFrame(values,columns=index1)

    # Concatenate all sensors data
    features = features.append(temp, ignore_index=True)

labels = pd.read_csv(main_dir + "Labels.csv", header=[0,1])
dataset = pd.concat([features, labels],  axis=1)
dataset.to_csv(main_dir + "dataset.csv")

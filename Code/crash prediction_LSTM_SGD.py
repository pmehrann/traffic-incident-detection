import numpy as np
from numpy import savetxt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import MDS, Isomap
from keras.models import Model, Sequential
from keras.utils import to_categorical
from sklearn.model_selection import KFold
from keras.layers import Bidirectional, LSTM, Dropout, Dense, Activation, LeakyReLU, Input, Conv1D, MaxPooling1D, RepeatVector, Flatten
from keras.optimizers import adam, adadelta, SGD, RMSprop
from keras.preprocessing.sequence import TimeseriesGenerator
from imblearn.over_sampling import SMOTE, SVMSMOTE, BorderlineSMOTE 
from imblearn.combine import SMOTEENN, SMOTETomek


def result_model(model,x_train,y_train,x_test,y_test,len_series,num_epochs,batch_size,num_features):

    # Create time series
    generator = TimeseriesGenerator(x_train, y_train, length=len_series, reverse = True, batch_size=1)
    generator_test = TimeseriesGenerator(x_test, y_test, length=len_series, reverse = True, batch_size=1)

    class_weight1 = {0: 1., 1: 400.}
    steps1 = len(generator) // batch_size
    trained_model = model.fit_generator(generator, steps_per_epoch=steps1, epochs=num_epochs, validation_data=generator_test, class_weight=class_weight1)

    score = model.evaluate_generator(generator_test)

    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    # plot
    # plt.figure(3)
    #     # plt.plot(trained_model.history['binary_accuracy'], linewidth=3)
    #     # plt.plot(trained_model.history['val_binary_accuracy'], linewidth=3)
    #     # plt.plot(trained_model.history['loss'], linewidth=3)
    #     # plt.plot(trained_model.history['val_loss'], linewidth=3)
    #     # plt.ylabel('Loss & Accuracy')
    #     # plt.xlabel('Epochs')
    #     # plt.legend(['acc', 'val_acc', 'loss', 'val_loss'], loc=10)
    #     # plt.grid(color='b', linestyle='--', linewidth=0.5)
    #     # plt.show()

    test_data = []
    true_labels = []
    for k in range(len(generator_test)):
      x, y = generator_test[k]
      test_data.append(x)
      true_labels.append(y)

    test_data = np.array(test_data)
    test_data = test_data.reshape(len(test_data),len_series,num_features)
    true_labels = np.array(true_labels)
    true_labels = true_labels.reshape(len(true_labels))

    predicted_y = model.predict(test_data)
    predicted_crash = np.around(predicted_y)

    return predicted_crash, true_labels


def confusion_matrix(predicted_labels,true_labels,x_test,flex):

    # Confusion Matrix
    con_mat = np.zeros((2, 2))
    for j in range(len(predicted_labels)):
        if (true_labels[j] == 1) and (sum(predicted_labels[j - flex:j + flex]) > 0):
            con_mat[0, 0] = con_mat[0, 0] + 1
        elif (true_labels[j] == 1) and (sum(predicted_labels[j - flex:j + flex]) == 0):
            con_mat[0, 1] = con_mat[0, 1] + 1
        elif (true_labels[j] == 0) and (predicted_labels[j] == 0):
            con_mat[1, 1] = con_mat[1, 1] + 1
        elif (true_labels[j] == 0) and (predicted_labels[j] == 1) and (sum(true_labels[j - flex:j + flex]) > 0):
            con_mat[1, 1] = con_mat[1, 1] + 1
        elif (true_labels[j] == 0) and (predicted_labels[j] == 1) and (sum(true_labels[j - flex:j + flex]) == 0):
            con_mat[1, 0] = con_mat[1, 0] + 1

    len_test = len(x_test)
    print("length of test data:", len_test)
    if (sum(sum(con_mat)) != len_test):
        print("invalid confusion matrix")

    return con_mat


def resampling(x_train,y_train,resampling):

    print("resampling training data using SVMSMOTE:")
    x_train_res, y_train_res = resampling.fit_resample(x_train, y_train.ravel())

    return x_train_res,y_train_res


def visualization(x_train,y_train,x_train_res,y_train_res,resampling):
    dir1 = "space_before_"+str(resampling)+".jpg"
    dir2 = "space_after_" + str(resampling) + ".jpg"

    # apply PCA to visualize feature space
    plt.figure(1)
    pca = PCA(n_components=2)
    x_train_pca = pca.fit_transform(x_train)
    crash = x_train_pca[y_train == 1]
    non_crash = x_train_pca[y_train == 0]
    plt.scatter(crash[:, 0], crash[:, 1])
    plt.scatter(non_crash[:, 0], non_crash[:, 1], c='r')
    plt.title('Feature space before resampling')
    plt.legend(['crash', 'non_crash'])
    plt.show()
    plt.savefig(dir1)

    plt.figure(2)
    pca = PCA(n_components=2)
    x_train_res_pca = pca.fit_transform(x_train_res)
    crash = x_train_res_pca[y_train_res == 1]
    non_crash = x_train_res_pca[y_train_res == 0]
    plt.scatter(crash[:, 0], crash[:, 1])
    plt.scatter(non_crash[:, 0], non_crash[:, 1], c='r')
    plt.title('Feature space after resampling')
    plt.legend(['crash', 'non_crash'])
    plt.show()
    plt.savefig(dir2)

    return


# load data
dataset = pd.read_csv('E:\MSc\Crash Detection\Data\Site1\Mix 5 min\site1_5min.csv', index_col=[0], header=[0,1])
dataset1 = pd.read_csv('E:\MSc\Crash Detection\Data\Site2\Mix 5 min\site2_5min.csv', index_col=[0], header=[0,1])

# cleaning the rows whose flow or occupancy equal -1
dataset=dataset[~(dataset[[('6368','Flow'),('6368','Occupancy'),('6369','Flow'),('6369','Occupancy'),('6370','Flow'),('6370','Occupancy'),
                           ('6371','Flow'),('6371','Occupancy'),('6372','Flow'),('6372','Occupancy'),('6373','Flow'),('6373','Occupancy'),
                           ('6374','Flow'),('6374','Occupancy'),('6375','Flow'),('6375','Occupancy'),('6426','Flow'),('6426','Occupancy'),
                           ('6427','Flow'),('6427','Occupancy'),('6428','Flow'),('6428','Occupancy'),('6429','Flow'),('6429','Occupancy'),
                           ('6430','Flow'),('6430','Occupancy'),('6431','Flow'),('6431','Occupancy'),('6432','Flow'),('6432','Occupancy'),
                           ('6433','Flow'),('6433','Occupancy')]].isin([-1])).any(1)]
dataset = dataset.reset_index(drop=True)

dataset1=dataset1[~(dataset1[[('6381','Flow'),('6381','Occupancy'),('6382','Flow'),('6382','Occupancy'),('6383','Flow'),('6383','Occupancy'),
                           ('6384','Flow'),('6384','Occupancy'),('6385','Flow'),('6385','Occupancy'),('6386','Flow'),('6386','Occupancy'),
                           ('6387','Flow'),('6387','Occupancy'),('6388','Flow'),('6388','Occupancy'),('6413','Flow'),('6413','Occupancy'),
                           ('6414','Flow'),('6414','Occupancy'),('6415','Flow'),('6415','Occupancy'),('6416','Flow'),('6416','Occupancy'),
                           ('6417','Flow'),('6417','Occupancy'),('6418','Flow'),('6418','Occupancy'),('6419','Flow'),('6419','Occupancy'),
                           ('6420','Flow'),('6420','Occupancy')]].isin([-1])).any(1)]
dataset1 = dataset1.reset_index(drop=True)

#cleaning the rows whose flow is greater than 3000
dataset=dataset[~(dataset[[('6368','Flow'),('6369','Flow'),('6370','Flow'),('6371','Flow'),('6372','Flow'),('6373','Flow'),('6374','Flow'),('6375','Flow'),
                           ('6426','Flow'),('6427','Flow'),('6428','Flow'),('6429','Flow'),('6430','Flow'),('6431','Flow'),('6432','Flow'),('6433','Flow')]].gt(3000)).any(1)]
dataset = dataset.reset_index(drop=True)

dataset1=dataset1[~(dataset1[[('6381','Flow'),('6382','Flow'),('6383','Flow'),('6384','Flow'),('6385','Flow'),('6386','Flow'),('6387','Flow'),('6388','Flow'),
                           ('6413','Flow'),('6414','Flow'),('6415','Flow'),('6416','Flow'),('6417','Flow'),('6418','Flow'),('6419','Flow'),('6420','Flow')]].gt(3000)).any(1)]
dataset1 = dataset1.reset_index(drop=True)

# separate labels and features and extract indexes of accidents
labels = dataset['labels']
labels = np.array(labels)
labels = labels.reshape(len(labels))
features = dataset.drop(columns='labels')

labels1 = dataset1['labels']
labels1 = np.array(labels1)
labels1 = labels1.reshape(len(labels1))
features1 = dataset1.drop(columns='labels')

features_all = np.concatenate([features,features1]) # train two sites together
labels_all = np.concatenate([labels,labels1])

# normalize data
scalar = preprocessing.StandardScaler()
features_all = scalar.fit_transform(features_all)

num_features = features_all.shape[1]
print("number of features after dimension reduction:",num_features)

# split the dataset using k-fold cross validation
kf = KFold(n_splits=5, random_state=None, shuffle=False)

x_dataset = []
y_dataset = []
x_dataset_test = []
y_dataset_test = []
for train_index, test_index in kf.split(features_all):
     print("TRAIN:", train_index, "TEST:", test_index)
     x_train, x_test = features_all[train_index], features_all[test_index]
     y_train, y_test = labels_all[train_index], labels_all[test_index]
     x_dataset.append(x_train)
     y_dataset.append(y_train)
     x_dataset_test.append(x_test)
     y_dataset_test.append(y_test)

x_dataset = np.array(x_dataset)
y_dataset = np.array(y_dataset)
x_dataset_test = np.array(x_dataset_test)
y_dataset_test = np.array(y_dataset_test)

#print('Number of non-crash data before oversampling in training data:', sum(y_train == 0))
#print('Number of crash data before oversampling in training data: {}\n'.format(sum(y_train == 1)))
#print('Number of non-crash data in test data:', sum(y_test == 0))
#print('Number of crash data in test data: {}\n'.format(sum(y_test == 1)))

sm1 = SMOTE()
sm2 = SVMSMOTE()
sm3 = SMOTEENN()
sm4 = SMOTETomek()
dict_res = dict([('smote', sm1), ('svmsmote', sm2), ('smoteenn', sm3), ('smotetomek', sm4)])
names_res = np.array(['smote', 'svmsmote', 'smoteenn', 'smotetomek'])

flex = np.array([1,6,12])
len_series = np.array([2,4,6,8])

for i in range(5):
    x_train = x_dataset[i]
    y_train = y_dataset[i]
    x_test = x_dataset_test[i]
    y_test = y_dataset_test[i]
    for j in range(4):

            # resampling training data
            x_train_res, y_train_res = resampling(x_train, y_train, dict_res[names_res[j]])

            # visualize resampled feature space
            visualization(x_train, y_train, x_train_res, y_train_res, names_res[j])
            for length in len_series:

                    # build the model
                    model = Sequential()
                    model.add(LSTM(100, return_sequences=False, activation=LeakyReLU(0.5), dropout=0.8, input_shape=(length, num_features)))

                    model.add(Dense(150, activation=LeakyReLU(0.5)))
                    model.add(Dropout(0.6))

                    model.add(Dense(200, activation=LeakyReLU(0.5)))
                    model.add(Dropout(0.5))

                    model.add(Dense(1, activation='sigmoid'))

                    # Compile the model
                    model.compile(optimizer=SGD(learning_rate=0.00001), loss='binary_crossentropy',
                                  metrics=['binary_accuracy'])
                    model.summary()

                    predicted_labels,true_labels = result_model(model, x_train, y_train, x_test, y_test, length, num_epochs=1, batch_size=64, num_features=num_features)
                    for f in flex:
                            con_mat = confusion_matrix(predicted_labels,true_labels,x_test,f)
                            print("confusion matrix of {}-th fold using resampling method {} with sequence length of {} with flexiblity {}:\n".format(i,names_res[j],length,f), con_mat)


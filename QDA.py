import os
import numpy as np
from collections import defaultdict

def get_train_dataset():

    data_dict = defaultdict(list)  # defaultdict(list),will build a dictionary with a default value of list
    for i in os.listdir("E:/dataset1/training_validation"):
        row = []
        label = int(i.split("_")[1])
        with open("E:/dataset1/training_validation/" + i) as file:
            for line in file:
                num = 0
                a = line.rstrip()
                for j in a:
                    if j is '1': num = num + 1
                row.append(num)

        data_dict[label].append(row)

    return data_dict

def get_test_dataset():

    data_dict = defaultdict(list)
    for i in os.listdir("E:/dataset1/test"):
        row = []
        label = int(i.split("_")[1])
        with open("E:/dataset1/test/" + i) as file:
            for line in file:
                num = 0
                a = line.rstrip()
                for j in a:
                    if j is '1': num = num + 1
                row.append(num)

        data_dict[label].append(row)

    return data_dict

def get_traindata_mean():
    class_mean = []
    for k in range(10):
        data=np.array(traindata[k])
        mean=data.sum(axis=0)/len(traindata[k])
        class_mean.append(mean)
    return class_mean



def get_covariance_matrix():
    cov_matrices = []
    for k in range(10):
        data=np.array(traindata[k])
        #class_mean=class_mean[k].reshape((1,32))
        data_norm=data-class_mean[k]
        data_norm=np.mat(data_norm)
        cov_matrix=data_norm.T * data_norm / (len(traindata[k])-1)
        cov_matrices.append(cov_matrix)
    return cov_matrices


def QDA(x):
    y_list = []
    for k in range(10):
        x=np.array(x)
        m = np.mat(x - class_mean[k])
        cov_matrix_inv = np.linalg.pinv(cov_matrices[k])
        d = np.log(np.linalg.det(cov_matrices[k]) + (1e-6)) # log(|sigma|) in case |sigma|=0 plus 1e-6
        y = -0.5 * d - 0.5 * m * cov_matrix_inv * m.T + np.log(len(traindata[k]) / 1934)
        #y=np.float(y)
        y_list.append(y)
    k_predict = y_list.index(max(y_list))
    return k_predict

traindata=get_train_dataset()
testdata=get_test_dataset()
class_mean=get_traindata_mean()
cov_matrices=get_covariance_matrix()

count=0
for k in range(10):
    for i in range(len(traindata[k])):
        x=traindata[k][i]
        y_predict=QDA(x)
        if y_predict == k:
            count += 1

acc=count/len(traindata)
print("train accuracy:",acc)

count1=0
for k in range(10):
    for i in range(len(testdata[k])):
        x=testdata[k][i]
        y_predict=QDA(x)
        if y_predict == k:
            count1 += 1

acc1=count1/len(testdata)
print("test accuracy:",acc1)

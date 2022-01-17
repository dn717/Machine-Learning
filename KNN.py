import os
import numpy as np

traindata=[]
label=[]
for i in os.listdir("E:/dataset1/training_validation"):
    tmp=[int(i.strip()) for i in open("E:/dataset1/training_validation/"+i).readlines()]
    traindata.append(tmp)
    label.append(i.split("_")[1])
#     break
traindata=np.array(traindata)
label=np.array(label)

testdata=[]
testlabel=[]
for i in os.listdir("E:/dataset1/test"):
    tmp=[int(i.strip()) for i in open("E:/dataset1/test/"+i).readlines()]
    testdata.append(tmp)
    testlabel.append(i.split("_")[1])
testdata=np.array(testdata)
testlabel=np.array(testlabel)

#KNN
import operator


def knn(trainData, testData, labels, k):
    rowSize = trainData.shape[0]
    diff = np.tile(testData, (rowSize, 1)) - trainData
    sqrDiff = diff ** 2
    sqrDiffSum = sqrDiff.sum(axis=1)
    distances = sqrDiffSum ** 0.5
    sortDistance = distances.argsort() #Sort the resulting distance from low to high

    count = {}
    #     print (labels)
    for i in range(k):
        vote = labels[sortDistance[i]]
        count[vote] = count.get(vote, 0) + 1
    sortCount = sorted(count.items(), key=operator.itemgetter(1), reverse=True) #Sort the frequency of occurrence of the classes from high to low

    return sortCount[0][0] #Return the most frequent class


from tqdm import tqdm

bestacc = 0
bestk = 1
k_iter = []
acc_iter = []
fold_iter = []
for k in range(1, 12):
    shuffle = np.random.permutation(len(traindata))
    traindata = traindata[shuffle]
    label = label[shuffle]
    for kfold in range(5):
        tdata = np.concatenate([traindata[:int(len(traindata) * (kfold / 5))], traindata[int(len(traindata) * ((kfold + 1) / 5)):]])
        tlabel = np.concatenate([label[:int(len(traindata) * (kfold / 5))], label[int(len(traindata) * ((kfold + 1) / 5)):]])
        valdata = traindata[int(len(traindata) * (kfold / 5)):int(len(traindata) * ((kfold + 1) / 5))]
        vallabel = label[int(len(traindata) * (kfold / 5)):int(len(traindata) * ((kfold + 1) / 5))]
        prelabel = []
        for i in tqdm(range(len(valdata))):
            #             print (i)
            prelabel.append(knn(tdata, valdata[i], tlabel, k))
        #         print (vallabel==prelabel)
        tmpacc = (sum((prelabel == vallabel).astype(int)) / len(prelabel))
        acc_iter.append(tmpacc)
        k_iter.append(k)
        fold_iter.append(kfold)
        if tmpacc > bestacc:
            bestacc = tmpacc
            bestk = k
        print("k=%d validation acc:" %k,sum((prelabel == vallabel).astype(int)) / len(prelabel))

print("the best k value is: ", bestk)
print("the best validation acc is: ", bestacc)

# %% Test

prelabel = []
for i in tqdm(range(len(testdata))):
    #             print (i)
    prelabel.append(knn(traindata, testdata[i], label, bestk))
testacc = (sum((prelabel == testlabel).astype(int)) / len(prelabel))
print("testacc:", testacc)

# %%

import pandas as pd

df = pd.DataFrame()
df["k"] = k_iter
df["acc"] = acc_iter
df["fold"] = fold_iter
print(df)
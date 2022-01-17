# -*- coding: utf-8 -*-
"""PCA"""

import numpy as np

traindata=np.load('/content/sample_data/traindata.npy')
testdata=np.load('/content/sample_data/testdata.npy')
trainlabel=np.load('/content/sample_data/trainlabel.npy')
testlabel=np.load('/content/sample_data/testlabel.npy')

n_sample,n_feature=traindata.shape

#Center the data(subtract the mean)
mean_vector=np.mean(traindata,axis=0)
traindata_norm=traindata-mean_vector
mean_vector_t=np.mean(testdata,axis=0)
testdata_norm=testdata-mean_vector_t

#compute the covariance matrix
arr=np.transpose(traindata)
cov_mat=np.cov(arr)
U,S,V=np.linalg.svd(cov_mat)
#another way to compute covariance matrix
#cov_mat = (1/(n_sample-1)) * np.dot(traindata_norm.transpose(),traindata_norm)

#get the k-dimension value when retain 95% variance
s=0
SUM=S.sum(dtype=float)*0.95
for n in S:
  s=s+n
  if s >= SUM:
    break

print(np.argwhere(S==n))
k=int(np.argwhere(S==n))+1 #k is the dimension


def svd_flip(u, v):
    """To ensure deterministic output from SVD.
    Adjusts the columns of U and the rows of V such that the loadings in the
    columns in u that are largest in absolute value are always positive,
    Same as the corresponding postion of v
    -------
    Returns
    -------
    u_adjusted, v_adjusted : arrays with the same dimensions as the input.
    """
    # columns of u, rows of v
    max_abs_cols = np.argmax(np.abs(u), axis=0)
    signs = np.sign(u[max_abs_cols, range(u.shape[1])])
    u *= signs
    v *= signs[:, np.newaxis]
    return u, v

#Using SVD decomposte the traindata_norm and get new dimension reduced data
U1,S1,V1=np.linalg.svd(traindata_norm,full_matrices=False)
U1,V1=svd_flip(U1,V1) #Ensure the Idempotence of SVD decomposition results
Z_train=np.dot(traindata_norm,V1[0:k,:].T) #V1.T is equal to the eigenvetor of covariance matrix,just it's ordered
print(Z_train)

#another way to get Z_train(using eigenvector of covariance matrix)
'''
a,b=np.linalg.eig(cov_mat)#a:eigenvalue, b:eigenvector
index = np.argsort(a) #sort a from max to min,return its index
U1=b[:,index[0:k]] #take corresponding index,take k=14 dimension(considering retain 95% variance of data)
Z_train=np.dot(traindata_norm,U1)
'''

# Apply same mapping to testdata_norm
U2,S2,V2=np.linalg.svd(testdata_norm,full_matrices=False)
U2,V2=svd_flip(U2,V2)
Z_test=np.dot(testdata_norm,V2[0:k,:].T)


"""Plot original Data in 3-dimension"""

U3,S3,V3=np.linalg.svd(traindata_norm)
Z=np.dot(traindata_norm,V3[0:3,:].T)
index_class=[0,189,387,582,781,967,1154,1349,1550,1730,1934]
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
ax = plt.subplot(projection='3d')
x=Z[:,0]
y=Z[:,1]
z=Z[:,2]
for c in range(10):
  ax.scatter(x[index_class[c]:index_class[c+1]], y[index_class[c]:index_class[c+1]], z[index_class[c]:index_class[c+1]])
  

ax.set_zlabel('z')
plt.show()

"""Apply data after using PCA  to the Logistic Regression"""

def train(data_arr, label_arr, n_class, iters = 1000, alpha = 0.1, lam = 0.01):
    '''
    @description: softmax train function
    @return: theta (parameters)
    '''    
    n_samples, n_features = data_arr.shape
    n_classes = n_class
    # Randomly initialize the weights matrix
    weights = np.random.rand(n_class, n_features)
    # define the lost result
    all_loss = list()
    # get one-hot matrix
    y_one_hot = one_hot(label_arr, n_samples, n_classes)
    for i in range(iters):
        scores = np.dot(data_arr, weights.T)
        # calculate softmax value
        probs = softmax(scores)
        # calculate the value of loss function
        loss = - (1.0 / n_samples) * np.sum(y_one_hot * np.log(probs))
        all_loss.append(loss)
        # calculate gradient
        dw = -(1.0 / n_samples) * np.dot((y_one_hot - probs).T, data_arr) + lam * weights
        dw[:,0] = dw[:,0] - lam * weights[:,0]
        # update weights matrix
        weights  = weights - alpha * dw
    return weights, all_loss
        

def softmax(scores):
    # calculate the sum 
    sum_exp = np.sum(np.exp(scores), axis = 1,keepdims = True)
    softmax = np.exp(scores) / sum_exp
    return softmax


def one_hot(label_arr, n_samples, n_classes):
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), label_arr.T] = 1
    return one_hot


def predict(test_dataset, label_arr, weights):
    scores = np.dot(test_dataset, weights.T)
    probs = softmax(scores)
    return np.argmax(probs, axis=1).reshape((-1,1))

label_arr = np.array(trainlabel).reshape((-1,1))
test_label_arr = np.array(testlabel).reshape((-1,1))

if __name__ == "__main__":
    
    #train
    weights, all_loss = train(Z_train,label_arr, n_class = 10)

    #test
    n_test_samples = testdata.shape[0]
    y_predict = predict(Z_test, test_label_arr, weights)
    accuray = np.sum(y_predict == test_label_arr) / n_test_samples
    print(accuray)

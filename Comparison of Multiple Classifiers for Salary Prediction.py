# -*- coding: utf-8 -*-
"""PR-Final Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Usd5V3IFSeV8bGYEB1ZCg9iNvEd2dkvr
"""

import pandas as pd
import numpy as np

data=pd.read_csv('/content/sample_data/adult.csv')
data

data.info()

data.columns=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','gain','loss','hours','country','income']

#data.rename(columns={'capital-gain': 'gain', 'capital-loss': 'loss', 'native-country': 'country','hours-per-week': 'hours'}, inplace=True)
data.columns

#Finding the special characters in the data frame 
data.isin(['?']).sum(axis=0)

# replace the special character to nan and then drop the nan rows
data['workclass'] = data['workclass'].replace('?',np.nan)
data['occupation'] = data['occupation'].replace('?',np.nan)
data['country']=data['country'].replace('?',np.nan)

#dropping the nan rows 
data.dropna(how='any',inplace=True)

n_samples=data.shape[0]
n_greater_50k=data[data['income'] =='>50K'].shape[0]
n_less_50k=data[data['income'] =='<=50K'].shape[0]

print('the number of income greater than 50k:',n_greater_50k)
print('the number of income less than 50k:',n_less_50k)

#data.iloc[:,[n]] the index=n+1  cloumn
#data.iloc[[n],:] the n+1 row

#normalization-scaling on numerical features
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
numerical_features=['age','fnlwgt','education-num','gain','loss','hours']
data[numerical_features]=scaler.fit_transform(data[numerical_features])

#dropping some useless features
data.drop(['country'], axis=1, inplace=True)

#mapping the data into numerical data using map function
data['income'] = data['income'].map({'<=50K': 0, '>50K': 1}).astype(int)

###---Way1:One-hot encoding on categorical features ---###
categorical_features=['workclass','education','marital-status','relationship','occupation','race','sex']
data=pd.get_dummies(data)

###---Way2:Exchange the value of categorical feature with number ---###
#sex
data['sex'] = data['sex'].map({'Male': 0, 'Female': 1}).astype(int)
#race
data['race'] = data['race'].map({'Black': 0, 'Asian-Pac-Islander': 1,'Other': 2, 'White': 3, 'Amer-Indian-Eskimo': 4}).astype(int)
#marital
data['marital-status'] = data['marital-status'].map({'Married-spouse-absent': 0, 'Widowed': 1, 'Married-civ-spouse': 2, 'Separated': 3, 'Divorced': 4,'Never-married': 5, 'Married-AF-spouse': 6}).astype(int)
#workclass
data['workclass'] = data['workclass'].map({'Self-emp-inc': 0, 'State-gov': 1,'Federal-gov': 2, 'Without-pay': 3, 'Local-gov': 4,'Private': 5, 'Self-emp-not-inc': 6}).astype(int)
#education
data['education'] = data['education'].map({'Some-college': 0, 'Preschool': 1, '5th-6th': 2, 'HS-grad': 3, 'Masters': 4, '12th': 5, '7th-8th': 6, 'Prof-school': 7,'1st-4th': 8, 'Assoc-acdm': 9, 'Doctorate': 10, '11th': 11,'Bachelors': 12, '10th': 13,'Assoc-voc': 14,'9th': 15}).astype(int)
#occupation
data['occupation'] = data['occupation'].map({ 'Farming-fishing': 1, 'Tech-support': 2, 'Adm-clerical': 3, 'Handlers-cleaners': 4, 
'Prof-specialty': 5,'Machine-op-inspct': 6, 'Exec-managerial': 7,'Priv-house-serv': 8,'Craft-repair': 9,'Sales': 10, 'Transport-moving': 11, 'Armed-Forces': 12, 'Other-service': 13,'Protective-serv':14}).astype(int)
#relationship
data['relationship'] = data['relationship'].map({'Not-in-family': 0, 'Wife': 1, 'Other-relative': 2, 'Unmarried': 3,'Husband': 4,'Own-child': 5}).astype(int)

#feature correlation analysis-use with WAY2
from scipy.stats import pointbiserialr,spearmanr
col_names=data.columns
feature=[]
correlation=[]
abs_correlation=[]
for c in col_names:
  if c  != 'income':
    if len(data[c].unique()) <= 2:
      corr=spearmanr(data['income'],data[c])[0]
    else:
      corr=pointbiserialr(data['income'],data[c])[0]
    feature.append(c)
    correlation.append(corr)
    abs_correlation.append(abs(corr))

feature_df=pd.DataFrame({'correaltion':correlation,'feature':feature,'abs_correlation':abs_correlation})
feature_df=feature_df.sort_values(by=['abs_correlation'],ascending=False)
feature_df=feature_df.set_index('feature')

feature_df

#heatmap of correlation matrix-use with Way2
import seaborn as sns 
import matplotlib.pyplot as pplt 
# use the heatmap function from seaborn to plot the correlation matrix
corrmat = data.corr()
f, ax = pplt.subplots(figsize=(12, 9))
k = 13 #number of variables for heatmap
cols = corrmat.nlargest(k, 'income')['income'].index
cor_matrix = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.5)
heat_map = sns.heatmap(cor_matrix, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
pplt.show()

final_features=list(data.columns)
print("{} total features after preprocessing".format(len(final_features)-1))

label=data['income']

data_without_label=data.drop(['income'],axis=1,inplace=False)

#split data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data_without_label,label,test_size=0.2)

print("number of training samples:{}".format(x_train.shape[0]))
print("number of testing samples:{}".format(x_test.shape[0]))

x_train=np.array(x_train)
x_test=np.array(x_test)
y_train=np.array(y_train)
y_test=np.array(y_test)

np.save('x_train.npy',x_train)
np.save('x_test.npy',x_test)
np.save('y_train.npy',y_train)
np.save('y_test.npy',y_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,precision_score,accuracy_score,recall_score,f1_score
from sklearn.metrics import classification_report

#Logistic Regression
LR=LogisticRegression()
LR.fit(x_train,y_train)
y_pred = LR.predict(x_test)

# #confusion matrix 
# cnf_matrix = confusion_matrix(y_test, y_pred)
# cnf_matrix
# TP=cnf_matrix[0][0]
# FN=cnf_matrix[0][1]
# FP=cnf_matrix[1][0]
# TN=cnf_matrix[1][1]
# recall=TP/(TP+FN)
# precision=TP/(TP+FP)
# F1_score=(2*precision*recall)/(precision+recall)
# print("recall:{:.4f},precision:{:.4f},F1-score={:.4f}".format(recall,precision,F1_score))

print ('-'*20+'LR'+'-'*20)
print ('Accuracy score:')
print (accuracy_score(y_test,y_pred))
print ('-'*40)
print ('Confusion Matrix:')
print (confusion_matrix(y_test,y_pred))
print ('-'*40)
print ('Classification Matrix:')
print (classification_report(y_test,y_pred))

#QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
clf=QuadraticDiscriminantAnalysis()
clf.fit(x_train,y_train)
y_pred2=clf.predict(x_test)

print ('-'*20+'QDA'+'-'*20)
print ('Accuracy score:')
print (accuracy_score(y_test,y_pred2))
print ('-'*40)
print ('Confusion Matrix:')
print (confusion_matrix(y_test,y_pred2))
print ('-'*40)
print ('Classification Matrix:')
print (classification_report(y_test,y_pred2))

#KNN
#5 fold Cross-Validation to get best k
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
#find the best k 
best_k=-1
best_score=0
for i in range(1,11):
        knn_clf=KNeighborsClassifier(n_neighbors=i)
        cv_scores = cross_val_score(knn_clf, x_train, y_train, cv=5)
        cv_scores_mean=np.mean(cv_scores)
        scores=cv_scores_mean
        if scores>best_score:
            best_score=scores
            best_k=i
print('The best k is:%d,The best score is:%.4f'%(best_k,best_score))

# #GridSearch way to get best k
# from sklearn.model_selection import GridSearchCV
# #create new a knn model
# knn_clf2 = KNeighborsClassifier()
# #create a dictionary of all values we want to test for n_neighbors
# param_grid = {???n_neighbors???: np.arange(1, 11)}
# #use gridsearch to test all values for n_neighbors cv:cross-validation
# knn_gscv = GridSearchCV(knn_clf2, param_grid, cv=5)

# #check top performing n_neighbors value
# knn_gscv.best_params_
# #fit model to data
# knn_gscv.fit(x_train,y_train)

knn_clf=KNeighborsClassifier(n_neighbors=10)
knn_clf.fit(x_train,y_train)
y_pred3=knn_clf.predict(x_test)
print ('-'*20+'KNN'+'-'*20)
print ('Accuracy score:')
print (accuracy_score(y_test,y_pred3))
print ('-'*40)
print ('Confusion Matrix:')
print (confusion_matrix(y_test,y_pred3))
print ('-'*40)
print ('Classification Matrix:')
print (classification_report(y_test,y_pred3))

#Gussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred4 = gnb.fit(x_train, y_train).predict(x_test)
print ('-'*20+'Naive Bayes'+'-'*20)
print ('Accuracy score:')
print (accuracy_score(y_test,y_pred4))
print ('-'*40)
print ('Confusion Matrix:')
print (confusion_matrix(y_test,y_pred4))
print ('-'*40)
print ('Classification Matrix:')
print (classification_report(y_test,y_pred4))

#USE PCA
from sklearn.decomposition import PCA
pca = PCA(0.95) #retain 95% variance
pca.fit(x_train)

print( pca.n_components_) #how many features PCA choose

z_train=pca.transform(x_train)
z_test=pca.transform(x_test)

#LR-PCA
LR=LogisticRegression()
LR.fit(z_train,y_train)
y_pred_pca = LR.predict(z_test)

print ('-'*20+'LR-PCA'+'-'*20)
print ('Accuracy score:')
print (accuracy_score(y_test,y_pred_pca))
print ('-'*40)
print ('Confusion Matrix:')
print (confusion_matrix(y_test,y_pred_pca))
print ('-'*40)
print ('Classification Matrix:')
print (classification_report(y_test,y_pred_pca))

#QDA-PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
clf=QuadraticDiscriminantAnalysis()
clf.fit(z_train,y_train)
y_pred_pca2=clf.predict(z_test)

print ('-'*20+'QDA-PCA'+'-'*20)
print ('Accuracy score:')
print (accuracy_score(y_test,y_pred_pca2))
print ('-'*40)
print ('Confusion Matrix:')
print (confusion_matrix(y_test,y_pred_pca2))
print ('-'*40)
print ('Classification Matrix:')
print (classification_report(y_test,y_pred_pca2))

#KNN-PCA
from sklearn.neighbors import KNeighborsClassifier
knn_clf=KNeighborsClassifier(n_neighbors=10)
knn_clf.fit(z_train,y_train)
y_pred3_pca=knn_clf.predict(z_test)
print ('-'*20+'KNN-PCA'+'-'*20)
print ('Accuracy score:')
print (accuracy_score(y_test,y_pred3_pca))
print ('-'*40)
print ('Confusion Matrix:')
print (confusion_matrix(y_test,y_pred3_pca))
print ('-'*40)
print ('Classification Matrix:')
print (classification_report(y_test,y_pred3_pca))

#Naive Bayes-PCA
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred4_pca = gnb.fit(z_train, y_train).predict(z_test)

print ('-'*20+'Naive Bayes-PCA'+'-'*20)
print ('Accuracy score:')
print (accuracy_score(y_test,y_pred4_pca))
print ('-'*40)
print ('Confusion Matrix:')
print (confusion_matrix(y_test,y_pred4_pca))
print ('-'*40)
print ('Classification Matrix:')
print (classification_report(y_test,y_pred4_pca))
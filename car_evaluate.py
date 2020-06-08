# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 19:55:59 2020

@author: kashe
"""
from IPython import get_ipython
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.model_selection import KFold,GridSearchCV,cross_val_score,train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier,BaggingClassifier

get_ipython().run_line_magic('matplotlib','qt')


names = ['Buying','Mantainence','Doors','Persons','Lug Boot','Safety','Class']
df = pd.read_csv('car-data.csv',names=names)





for col in df.columns:
    encoder = LabelEncoder().fit(df[col])
    df[col] = encoder.transform(df[col])
    
array = df.values
x = array[:,0:6]
y = array[:,6]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=7)
  
models = []
models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('KN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))

results=[]
names=[]

for name,model in models:
    
    kfold = KFold(n_splits=10,random_state=7,shuffle=True)
    cv_result = cross_val_score(model,x_train,y_train,cv=kfold,scoring='accuracy') 
    results.append(cv_result)
    names.append(name)
    
    msg = "%s: %f %f"%(name,cv_result.mean()*100,cv_result.std())
    print(msg)
    

pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',
LogisticRegression(solver='liblinear'))])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA',
LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',
KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',
DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB',
GaussianNB())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM',
SVC(gamma='auto'))])))
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std())
    print(msg)
    

model= SVC(gamma='auto')
scaler = StandardScaler().fit(x_train)
rescaledx = scaler.transform(x_train)
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
kfold = KFold(n_splits=10,random_state=7,shuffle=True)
param_grid = dict(C=c_values,kernel=kernel_values)
grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring='accuracy')
grid_result = grid.fit(rescaledx,y_train)
print(grid_result.best_score_,grid_result.best_params_)


model = KNeighborsClassifier()
scaler = StandardScaler().fit(x_train)
rescaledx = scaler.transform(x_train)
n_neigb = [1,3,5,7,9,11,13,15,17,19,21]
param_grid = dict(K=n_neigb)
kfold = KFold(n_splits=10,random_state=7,shuffle=True)
param_grid = dict(n_neighbors=n_neigb)
grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring='accuracy')
grid_result = grid.fit(rescaledx,y_train)
print(grid_result.best_score_,grid_result.best_params_)

'''
predictor = KNeighborsClassifier(n_neighbors=9)
predictor = SVC(C=0.9,kernel='rbf')
scaler = StandardScaler().fit(x_train)
rescaledx = scaler.transform(x_train)
rescaled_test = scaler.transform(x_test)
predictor.fit(rescaledx,y_train)
predictions = predictor.predict(rescaled_test)
print(accuracy_score(y_test,predictions)*100)
'''
    

models = []
models.append(('DT',DecisionTreeClassifier()))
models.append(('RT',RandomForestClassifier()))
models.append(('AC',AdaBoostClassifier()))
models.append(('ET',ExtraTreeClassifier()))
models.append(('GT',GradientBoostingClassifier()))

names=[]
result=[]

for name,model in models:
    kfold = KFold(n_splits=10,random_state=7,shuffle=True)
    cv_result = cross_val_score(model,x_train,y_train,cv=kfold,scoring='accuracy')
    names.append(name)
    result.append(cv_result)
    msg = '%s: %f (%f)'%(name,cv_result.mean()*100,cv_result.std())
    print(msg)
    

pipeline = []
pipeline.append(('ScaledDT',Pipeline([('Scale',StandardScaler()),('DT',DecisionTreeClassifier())])))
pipeline.append(('ScaledRT',Pipeline([('Scale',StandardScaler()),('RT',RandomForestClassifier())])))
pipeline.append(('ScaledAC',Pipeline([('Scale',StandardScaler()),('AC',AdaBoostClassifier())])))
pipeline.append(('ScaledET',Pipeline([('Scale',StandardScaler()),('ET',ExtraTreeClassifier())])))
pipeline.append(('ScaledGT',Pipeline([('Scale',StandardScaler()),('GT',GradientBoostingClassifier())])))

for name,model in pipeline:
    kfold = KFold(n_splits=10,random_state=7,shuffle=True)
    cv_result = cross_val_score(model,x_train,y_train,cv=kfold,scoring='accuracy')
    names.append(name)
    result.append(cv_result)
    msg = '%s: %f (%f)'%(name,cv_result.mean()*100,cv_result.std())
    print(msg)
 
model = GradientBoostingClassifier()
scaler = StandardScaler().fit(x_train)
rescaledx = scaler.transform(x_train)
n_estimators = [50,100,150,200,250,300,350,400,450]
param_grid = dict(n_estimators=n_estimators)
kfold = KFold(n_splits=10,random_state=7,shuffle=True)
grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring='accuracy')
grid_result = grid.fit(rescaledx,y_train)
print(grid_result.best_score_,grid_result.best_params_)


predictor = GradientBoostingClassifier(n_estimators=450)

scaler = StandardScaler().fit(x_train)
rescaledx = scaler.transform(x_train)
rescaled_test = scaler.transform(x_test)
predictor.fit(rescaledx,y_train)
predictions = predictor.predict(rescaled_test)
print(accuracy_score(y_test,predictions)*100)

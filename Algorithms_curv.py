# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:47:40 2019

@author: User
"""

import numpy as np
import random
#from open3d import *
import copy
import matplotlib.pyplot as plt
from scipy.spatial import distance
import math
import statistics
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import pandas
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV
#from registration import *
np.random.seed(1)
random.seed(1)

def preprocess(data):
    #data type should be list
    #length of the dataset
    length = len(data)
    index = range(length)
    
    #index of data for training
    trainindex = random.sample(index,int(4*length/5)) 
    
    #datasets for model parameter selection
    n = 5
    data_sel = random.sample([data[i] for i in trainindex],n)
    #print(len(data_sel))
    data_select = data_sel[0]
    for i in data_sel[1:]:
        data_select = np.concatenate((data_select,i),axis = 0)
    
    data_select = np.asarray(data_select)
    data_select = data_select[:,1:]
    
    #index of data for testing
    testindex = []
    for m in index:
        if m not in trainindex:
            testindex.append(m)
    
    return data_select, trainindex, testindex


def test_model(data,testindex,model):
    #F-measure result list
    F_measure_list = []
    
    #G-mean result list
    G_mean_list = []
    
    #get the F-measure and G-mean result of all the data from testindex
    for i in testindex:
        #manage test dataset
        test = data[i]
        test = np.asarray(test)
        test = test[:,1:]
        X_test = test[:,(3,4,5,6,7)]
        y_test = test[:,-1]
        y_test = list(y_test)
        
        #start test
        if y_test.count(1)<1:
            break
        else:
            #prediction result
            pred = model.predict(X_test)
            
            #define defect with label 1 as positive, the other class as negative
            TP = 0 #True Positive
            FP = 0 #False Positive
            TN = 0 #True Negative
            FN = 0 #False Negative
            
            for j in range(len(y_test)):
                if y_test[j] == 0 and y_test[j] == pred[j]:
                    TN = TN + 1
                if y_test[j] == 1 and y_test[j] == pred[j]:
                    TP = TP + 1    
                if y_test[j] == 0 and y_test[j] != pred[j]:
                    FN = FN + 1  
                if y_test[j] == 1 and y_test[j] != pred[j]:
                    FP = FP + 1
            
            #calculate True Positive rate and True Negative rate
            if TP+FN==0:
                TP_rate = 0
            else:
                TP_rate = TP/(TP+FN) #True positive rate
            if TN+FP==0:
                TN_rate = 0
            else:
                TN_rate = TN/(TN+FP) #True negative rate
            
            #Positive predictive value
            PP_value = TP/(TP+FP) 
            
            #F-measure
            R = TP_rate
            P = PP_value
            if R == 0 or P == 0:
                F_measure = 0
            if R != 0 and P != 0:
                F_measure = 2/(1/R+1/P)
            #print(F_measure)
            F_measure_list.append(F_measure)
            #F-measure average
            F_ave = sum(F_measure_list)/len(F_measure_list)
            #F-measure std
            F_std = np.std(F_measure_list)
            
            #G-mean
            G_mean = math.sqrt(TP_rate*TN_rate)
            #print(G_mean)
            G_mean_list.append(G_mean)
            #G-mean average
            G_ave = sum(G_mean_list)/len(G_mean_list)
            #F-measure std
            G_std = np.std(G_mean_list)
            
    return F_ave, F_std, G_ave, G_std        
    

def print_result(F_ave,F_std,G_ave,G_std):
    #mean F-measure
    print("Mean F-measure is: ", F_ave)
    
    #standard deviation
    print("Standard deviation of F-measure is: ",F_std)
    
    # mean G-mean
    print("Mean G-mean is: ",G_ave)
    
    #standard deviation
    print("Standard deviation of G-mean is: ",G_std)    


def Bagging(data):
    np.random.seed(1)
    random.seed(1)
    #load data
    data_select, trainindex, testindex = preprocess(data)
    
    ##hyperparameters tuning
    #load the data for tuning
    X = data_select[:,(3,4,5,6,7)]
    X = pandas.DataFrame(X)
    y = data_select[:,-1]
    y = np.asarray(y)
    y_list = list(y)
    # use SMOTE to oversampling the dataset
    if y_list.count(1)>7: #request for smote 
        sm = SMOTE()
        newX, newy = sm.fit_sample(X, y)
    else:
        ros = RandomOverSampler()
        newX, newy= ros.fit_resample(X, y)
    
    #tuning the selected parameters
    #define the parameters range for selection
    #estimators = [10,100,1000]
    #max_samples = [0.1, 0.5, 0.9]
    #max_features = [0.1, 0.5, 0.9]
    
    #use function GridSearchCV to find the best parameters values
    #nfolds = 5
    #param_grid = {'n_estimators': estimators, 'max_samples' : max_samples, 
    #              'max_features':max_features}
    #grid_search = GridSearchCV(BaggingClassifier(), param_grid, cv=nfolds)
    #grid_search.fit(newX, newy)
    #print(grid_search.best_params_)
    
    
    ##Train the model
    #define the model
    #model = BaggingClassifier(n_estimators=grid_search.best_params_['n_estimators'],
    #                          max_features = grid_search.best_params_['max_features'],
    #                          max_samples = grid_search.best_params_['max_samples'],
    #                          oob_score = True)
    
    model = BaggingClassifier()

    #train the model with datasets from 1 to 40
    #trainindex = random.sample(index,int(4*length/5))
    for i in trainindex:
        patch1 = data[i]
        patch1 = np.asarray(patch1)
        #organize the training data
        patch1 = patch1[:,1:]
        #print(patch1)
        np.random.shuffle(patch1)
        #n = int(0.8*len(patch1))
        x_train = patch1[:,(3,4,5,6,7)] #add second derovative in x axis of each point as input]
        y_train = patch1[:,-1]
        y_train_list = list(y_train)
        if y_train_list.count(1)>=6: #request for smote 
            sm = SMOTE()
            newx, newy = sm.fit_sample(x_train, y_train)
            model.fit(newx,newy)
        elif y_train_list.count(1) == 0:
            continue
        elif y_train_list.count(1) > 0 and y_train_list.count(1)<6:
            ros = RandomOverSampler()
            newx, newy= ros.fit_resample(x_train, y_train)
            model.fit(newx,newy)
    
    #Test with test datasets
    F_ave, F_std, G_ave, G_std = test_model(data,testindex,model)
    
    #print the result
    print("Bagging results:")
    print_result(F_ave,F_std,G_ave,G_std)
    return model,F_ave, F_std, G_ave, G_std
    
    
def gradient_boosting(data):
    np.random.seed(1)
    random.seed(1)
    #load data
    data_select, trainindex, testindex = preprocess(data)
    
    ##hyperparameters tuning
    #load the data for tuning
    X = data_select[:,(3,4,5,6,7)]
    X = pandas.DataFrame(X)
    print("X: ",X.head())
    y = data_select[:,-1]
    y = np.asarray(y)
    y_list = list(y)
    
    # use SMOTE to oversampling the dataset
    if y_list.count(1)>7: #request for smote 
        sm = SMOTE()
        newX, newy = sm.fit_sample(X, y)
    else:
        ros = RandomOverSampler()
        newX, newy= ros.fit_resample(X, y)
    
    #tuning the selected parameters
    #define the parameters range for selection
    #estimators = [10,100,500]
    #learnrates = [0.01, 0.1, 0.5]
    #depths = [1,10,30]
    
    #use function GridSearchCV to find the best parameters values
    #nfolds = 5
    #param_grid = {'n_estimators': estimators, 'learning_rate' : learnrates, 
    #              'max_depth':depths}
    #grid_search = GridSearchCV(GradientBoostingClassifier(), param_grid, 
    #                           cv=nfolds)
    #grid_search.fit(newX, newy)
    #print(grid_search.best_params_)
    
    ##Train the model
    #define the model
    #model = GradientBoostingClassifier(n_estimators=grid_search.best_params_['n_estimators'], 
    #                                   learning_rate=grid_search.best_params_['learning_rate'],
    #                                   max_depth=grid_search.best_params_['max_depth'])
    
    model = GradientBoostingClassifier()

    #train the model with datasets from 1 to 40
    #trainindex = random.sample(index,int(4*length/5))
    for i in trainindex:
        patch1 = data[i]
        patch1 = np.asarray(patch1)
        #organize the training data
        patch1 = patch1[:,1:]
        #print(patch1)
        np.random.shuffle(patch1)
        #n = int(0.8*len(patch1))
        x_train = patch1[:,(3,4,5,6,7)] 
        y_train = patch1[:,-1]
        y_train_list = list(y_train)
        if y_train_list.count(1)>7: #request for smote 
            sm = SMOTE()
            newx, newy = sm.fit_sample(x_train, y_train)
            model.fit(newx,newy)
        else:
            ros = RandomOverSampler()
            newx, newy= ros.fit_resample(x_train, y_train)
            model.fit(newx,newy)
    
    #Test with test datasets
    F_ave, F_std, G_ave, G_std = test_model(data,testindex,model)
    
    #print the result
    print("Gradient result:")
    print_result(F_ave,F_std,G_ave,G_std)
    return model,F_ave, F_std, G_ave, G_std
    
    
def random_forest(data):
    np.random.seed(1)
    random.seed(1)
    #load data
    data_select, trainindex, testindex = preprocess(data)
    
    ##hyperparameters tuning
    #load the data for tuning
    X = data_select[:,(3,4,5,6,7)]
    X = pandas.DataFrame(X)
    y = data_select[:,-1]
    y = np.asarray(y)
    y_list = list(y)
    
    # use SMOTE to oversampling the dataset
    if y_list.count(1)>7: #request for smote 
        sm = SMOTE()
        newX, newy = sm.fit_sample(X, y)
    else:
        ros = RandomOverSampler()
        newX, newy= ros.fit_resample(X, y)
    
    #tuning the selected parameters
    #define the parameters range for selection
    #estimators = [10,100,1000]
    #min_samples_leaf = [0.01, 0.1, 0.5]
    #min_samples_split = [0.01,0.1,0.5]
    #min_weight_fraction_leaf = [0.01,0.1,0.5]
    
    #use function GridSearchCV to find the best parameters values
    #nfolds = 5
    #param_grid = {'n_estimators': estimators, 
    #              'min_samples_leaf' : min_samples_leaf, 
    #              'min_samples_split':min_samples_split,
    #              'min_weight_fraction_leaf':min_weight_fraction_leaf}
    #grid_search = GridSearchCV(RandomForestClassifier(class_weight = 'balanced_subsample'), 
    #                           param_grid, cv=nfolds)
    #grid_search.fit(newX, newy)
    #print(grid_search.best_params_)
    
    ##Train the model
    #define the model
    #model = RandomForestClassifier(min_samples_leaf = grid_search.best_params_['min_samples_leaf'], 
    #                           min_samples_split = grid_search.best_params_['min_samples_split'],
    #                           min_weight_fraction_leaf = grid_search.best_params_['min_weight_fraction_leaf'], 
    #                           n_estimators = grid_search.best_params_['n_estimators'],
    #                           class_weight = 'balanced_subsample')

    model = RandomForestClassifier()
    
    #train the model with datasets from 1 to 40
    #trainindex = random.sample(index,int(4*length/5))
    for i in trainindex:
        
        patch1 = data[i]
        patch1 = np.asarray(patch1)
        #organize the training data
        patch1 = patch1[ :,1:]
        #print(patch1)
        np.random.shuffle(patch1)
        #n = int(0.8*len(patch1))
        x_train = patch1[:,(3,4,5,6,7)] 
        y_train = patch1[:,-1]
        y_train_list = list(y_train)
        if y_train_list.count(1)>7: #request for smote 
            sm = SMOTE()
            newx, newy = sm.fit_sample(x_train, y_train)
            model.fit(newx,newy)
        else:
            ros = RandomOverSampler()
            newx, newy= ros.fit_resample(x_train, y_train)
            model.fit(newx,newy)
    
    #Test with test datasets
    F_ave, F_std, G_ave, G_std = test_model(data,testindex,model)
    
    #print the result
    print("Random forest result:")
    print_result(F_ave,F_std,G_ave,G_std)
    return model,F_ave, F_std, G_ave, G_std
    
    
def linear_SVM(data):
    np.random.seed(1)
    random.seed(1)
    #load data
    data_select, trainindex, testindex = preprocess(data)
    
    ##hyperparameters tuning
    #load the data for tuning
    X = data_select[:,(3,4,5,6,7)]
    X = pandas.DataFrame(X)
    y = data_select[:,-1]
    y = np.asarray(y)
    y_list = list(y)
    
    # use SMOTE to oversampling the dataset
    if y_list.count(1)>7: #request for smote 
        sm = SMOTE()
        newX, newy = sm.fit_sample(X, y)
    else:
        ros = RandomOverSampler()
        newX, newy= ros.fit_resample(X, y)
    
    #tuning the selected parameters
    #define the parameters range for selection
    tolerance = [1e-5,1e-4,1e-3,1e-2,1e-1]
    Cs = [0.001, 0.01, 0.1, 1, 10]
    
    
    #use function GridSearchCV to find the best parameters values
    nfolds = 5
    param_grid = {'tol': tolerance, 'C' : Cs}
    grid_search = GridSearchCV(LinearSVC(dual=False, class_weight = 'balanced'), 
                               param_grid, cv=nfolds)
    grid_search.fit(newX, newy)
    print(grid_search.best_params_)
    
    ##Train the model
    #define the model
    model = LinearSVC(dual=False, class_weight = 'balanced', 
                      tol =grid_search.best_params_['tol'], 
                      C =grid_search.best_params_['C'])

    #model = LinearSVC()    

    #train the model with datasets from 1 to 40
    #trainindex = random.sample(index,int(4*length/5))
    for i in trainindex:
        print(i)
        patch1 = data[i]
        patch1 = np.asarray(patch1)
        #organize the training data
        patch1 = patch1[:,1:]
        #print(patch1)
        np.random.shuffle(patch1)
        #n = int(0.8*len(patch1))
        x_train = patch1[:,(3,4,5,6,7)] 
        y_train = patch1[:,-1]
        y_train_list = list(y_train)
        if y_train_list.count(1)>=6: #request for smote 
            sm = SMOTE()
            newx, newy = sm.fit_sample(x_train, y_train)
            model.fit(newx,newy)
        elif y_train_list.count(1) == 0:
            continue
        elif y_train_list.count(1) > 0 and y_train_list.count(1)<6:
            ros = RandomOverSampler()
            newx, newy= ros.fit_resample(x_train, y_train)
            model.fit(newx,newy)
    
    #Test with test datasets
    F_ave, F_std, G_ave, G_std = test_model(data,testindex,model)
    
    #print the result
    print("Linear_SVM result:")
    print_result(F_ave,F_std,G_ave,G_std)   
    return model,F_ave, F_std, G_ave, G_std
    
    
def K_nearest_neighbors(data):
    np.random.seed(1)
    random.seed(1)
    #load data
    data_select, trainindex, testindex = preprocess(data)
    
    ##hyperparameters tuning
    #load the data for tuning
    X = data_select[:,(3,4,5,6,7)]
    X = pandas.DataFrame(X)
    y = data_select[:,-1]
    y = np.asarray(y)
    y_list = list(y)
    
    # use SMOTE to oversampling the dataset
    if y_list.count(1)>7: #request for smote 
        sm = SMOTE()
        newX, newy = sm.fit_sample(X, y)
    else:
        ros = RandomOverSampler()
        newX, newy= ros.fit_resample(X, y)
    
    #tuning the selected parameters
    #define the parameters range for selection
    n_neighbors = [10,50,100,200]
    leaf_size = [10,30,50,70,90]
    p = [1,2]
    
    #use function GridSearchCV to find the best parameters values
    nfolds = 5
    param_grid = {'n_neighbors': n_neighbors, 'leaf_size' : leaf_size, 'p':p}
    grid_search = GridSearchCV(KNeighborsClassifier(weights = 'distance',algorithm = 'auto'), param_grid, cv=nfolds)
    grid_search.fit(newX, newy)
    print(grid_search.best_params_)
    
    ##Train the model
    #define the model
    model = KNeighborsClassifier(weights = 'distance',algorithm = 'auto',
                                 leaf_size = grid_search.best_params_['leaf_size'], 
                             n_neighbors = grid_search.best_params_['n_neighbors'],
                             p = grid_search.best_params_['p'])

    #model = KNeighborsClassifier()
    
    #train the model with datasets from 1 to 40
    #trainindex = random.sample(index,int(4*length/5))
    for i in trainindex:
        patch1 = data[i]
        patch1 = np.asarray(patch1)
        #organize the training data
        patch1 = patch1[:,1:]
        #print(patch1)
        np.random.shuffle(patch1)
        #n = int(0.8*len(patch1))
        x_train = patch1[:,(3,4,5,6,7)] 
        y_train = patch1[:,-1]
        y_train_list = list(y_train)
        if y_train_list.count(1)>7: #request for smote 
            sm = SMOTE()
            newx, newy = sm.fit_sample(x_train, y_train)
            model.fit(newx,newy)
        else:
            ros = RandomOverSampler()
            newx, newy= ros.fit_resample(x_train, y_train)
            model.fit(newx,newy)
    
    #Test with test datasets
    F_ave, F_std, G_ave, G_std = test_model(data,testindex,model)
    
    #print the result
    print("K-nearest neighbors result:")
    print_result(F_ave,F_std,G_ave,G_std)
    return model,F_ave, F_std, G_ave, G_std
    
    
    
    
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
import math
from sklearn.model_selection import KFold
import time
from sklearn import preprocessing
import matplotlib.pyplot as plt

def normalizer(data):
    numOfFeat = data.shape[1]
    for i in range(numOfFeat):
        data[:,i] = (data[:,i]-min(data[:,i])) /(max(data[:,i])-min(data[:,i]))
    return data

def Haddamard(x,y):
    if (x==y):
        return 0
    else:
        return 1

def Euclidean(x,y):
    return (x-y)**2

def Manhattan(x,y):
    return abs(x-y)

def MSE_RMSE_MAPE(predictions,targets):
    mse=0
    rmse=0
    mape=0
    for index,i in enumerate(predictions):
        mse+=(targets[index]-i)**2
        mape+=(abs(targets[index]-i)/targets[index])*100
    mse = mse/len(predictions)
    rmse = math.sqrt(mse)
    mape= mape/len(predictions)
    return mse,rmse,mape


def MyKnnImplement(testdata, traindata, categorical_feat_index=[0, 1, 2, 3], nvalue=1, metric="first"):
    predictions = []
    for i in testdata:
        listofdistances = []
        for z in traindata:
            dist = 0
            catdist = 0
            condist = 0
            for index, j in enumerate(i):
                if index in categorical_feat_index:
                    catdist += Haddamard(i[index], z[index])
                else:
                    if metric=="first":
                        condist += Euclidean(i[index], z[index])
                    else:
                        condist += Manhattan(i[index], z[index])
            if metric=="first":
                dist = catdist + math.sqrt(condist)
            else:
                dist = catdist + condist
            listofdistances.append((dist, z[-1]))
        kneighbors = sorted(listofdistances)
        totalweight = 0
        weight = 0
        for f in kneighbors[:nvalue]:
            totalweight += f[0] ** 2
            weight += f[1] * (f[0] ** 2)
        predictions.append(weight / totalweight)
    return predictions

def myKNeighborsRegressor(X_normalized):
    kf = KFold(n_splits=3)
    mses,rmses,mapes,times=0,0,0,0
    for train_index, test_index in kf.split(X_normalized):
        knn = KNeighborsRegressor()
        testdata = X_normalized[test_index,:]
        traindata = X_normalized[train_index,:]
        trainlabel = y[train_index]
        testlabel = y[test_index]
        knn.fit(traindata,trainlabel)
        start = time.time()
        predictions = knn.predict(testdata)
        end = time.time()
        mse,rmse,mape = MSE_RMSE_MAPE(predictions,testlabel)
        mses+=mse
        rmses+=rmse
        mapes+=mape
        times+=(end-start)
    print('Average MSE = {}, Average RMSE = {}, Average MAPE = {} , Average Time = {} '.format(mses/3,rmses/3,mapes/3,times/3 ))
        
        
def myDecisionTreeRegressor(X_normalized):
    kf = KFold(n_splits=3)
    mses,rmses,mapes,times=0,0,0,0
    for train_index, test_index in kf.split(X_normalized):
        dt = DecisionTreeRegressor()
        testdata = X_normalized[test_index,:]
        traindata = X_normalized[train_index,:]
        trainlabel = y[train_index]
        testlabel = y[test_index]
        dt.fit(traindata,trainlabel)
        start = time.time()
        predictions = dt.predict(testdata)
        end = time.time()
        mse,rmse,mape = MSE_RMSE_MAPE(predictions,testlabel)
        mses+=mse
        rmses+=rmse
        mapes+=mape
        times+=(end-start)
    print('Average MSE = {}, Average RMSE = {}, Average MAPE = {} , Average Time = {} '.format(mses/3,rmses/3,mapes/3,times/3 ))


def myBayesianRidge(X_normalized):
    kf = KFold(n_splits=3)
    mses,rmses,mapes,times=0,0,0,0
    for train_index, test_index in kf.split(X_normalized):
        br = BayesianRidge()
        testdata = X_normalized[test_index,:]
        traindata = X_normalized[train_index,:]
        trainlabel = y[train_index]
        testlabel = y[test_index]
        br.fit(traindata,trainlabel)
        start = time.time()
        predictions = br.predict(testdata)
        end = time.time()
        mse,rmse,mape = MSE_RMSE_MAPE(predictions,testlabel)
        mses+=mse
        rmses+=rmse
        mapes+=mape
        times+=(end-start)
    print('Average MSE = {}, Average RMSE = {}, Average MAPE = {} , Average Time = {} '.format(mses/3,rmses/3,mapes/3,times/3 ))

if __name__ == '__main__':
    garmentsdf = pd.read_csv("garments_worker_productivity.csv")
    garmentsdfnan = garmentsdf.fillna(value=garmentsdf.mean()) ## fill wip with mean value (actually every nan value with their columns mean)
    garmentsdfnan.loc[garmentsdfnan.department == 'finishing ', 'department'] = 'finishing' ## There are 'finishing ' and  'finishing' so I switch them
    categorical_feat = ["quarter", "department", "day", "team"] # categorical features
    garmentsdfnan = garmentsdfnan.drop(columns=["date", "idle_time", "idle_men", "no_of_style_change", "targeted_productivity"]) #dropped columns
    targetdata = garmentsdfnan["actual_productivity"].to_numpy()
    continousdata = garmentsdfnan.drop(columns=categorical_feat + ["actual_productivity"]).to_numpy()
    categoricaldata = garmentsdfnan[categorical_feat].to_numpy()
    continousdatanormalized = normalizer(continousdata) # normalizing the continous variables
    full_Data = np.concatenate((categoricaldata, continousdatanormalized), axis=1)
    full_Data = np.concatenate((full_Data, targetdata.reshape(-1, 1)), axis=1) # getting together all the data
    kf = KFold(n_splits=3)
    fig, axs = plt.subplots(4,figsize=(16,16))
    for i in ["first","second"]:
        mses,rmses,mapes,times=[],[],[],[]
        for n in range(2, 11):
            totalmse,totalrmse,totalmape,averagetime=0,0,0,0
            for train_index, test_index in kf.split(full_Data):
                testdata = full_Data[test_index, :-1]
                traindata = full_Data[train_index, :]
                start = time.time()
                predictions = MyKnnImplement(testdata, traindata, nvalue=n,metric=i)
                end = time.time()
                averagetime+=(end - start)
                targets = full_Data[test_index, -1]
                mse, rmse, mape = MSE_RMSE_MAPE(predictions, targets)
                totalmse+=mse
                totalrmse += rmse
                totalmape += mape
            mses.append(totalmse/3)
            rmses.append(totalrmse/3)
            mapes.append(totalmape/3)
            times.append(averagetime/3)
            print('KNN with n = {} and {} metric , Average MSE = {}, Average RMSE = {}, Average MAPE = {} , Average Time = {} '.format(n,i,totalmse/3,totalrmse/3,totalmape/3,averagetime/3 ))
        
        
        axs[0].plot(range(2, 11), mses)
        axs[0].set_title("MSE")
        axs[1].plot(range(2, 11), rmses)
        axs[1].set_title("RMSE")
        axs[2].plot(range(2, 11), mapes)
        axs[2].set_title("MAPE")
        axs[3].plot(range(2, 11), times)
        axs[3].set_title("Avg. Prediction Times")
        plt.legend(["Metric1", "Metric2"])
        
    garmentsdfe = pd.get_dummies(garmentsdfnan,columns=categorical_feat)
    X = garmentsdfe.drop(columns=["actual_productivity"]).to_numpy()
    y = garmentsdfe["actual_productivity"].to_numpy()
    normalizer = preprocessing.StandardScaler().fit(X)
    X_normalized = normalizer.transform(X)
    print("KNeighborsRegressor --> ")
    myKNeighborsRegressor(X_normalized)
    print("Bayesian_Ridge_Regressor --> ")
    myBayesianRidge(X_normalized)
    print("DecisionTreeRegressor --> ")
    myDecisionTreeRegressor(X_normalized)

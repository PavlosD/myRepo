# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 04:35:38 2019

@author: Paul
"""


import torch
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from sklearn.model_selection import train_test_split

print("\n==========Iterating for Hasy dataset===========\n")
df = pd.read_csv('hasy-data-labels.csv')
df = df[df.symbol_id >= 70]
df = df[df.symbol_id <= 80]
data_vect = []
targets = []

for index, row in df.iterrows():
    img = plt.imread(row['path'])
    img = img.flatten()
    data_vect.append(img)
    targets.append(row['latex'])

X_tr, X_te, y_tr, y_te = train_test_split(data_vect, targets, train_size=0.8, test_size=0.2)
y_tr = list(map(int, y_tr))
y_te = list(map(int, y_te))
y_test_var = np.var(y_te)



X_train = torch.from_numpy(np.array(X_tr, dtype='float32'))
y_train = torch.from_numpy(np.array(y_tr, dtype='int_'))
X_test = torch.from_numpy(np.array(X_te, dtype='float32'))
y_test = torch.from_numpy(np.array(y_te, dtype='int_'))


mintrain=10000
mintest=10000

for epochs in range(20,121,20):
    for hidden_nodes in range(5,21,5):
        for learning_rate in range(1,10,2):
            print("\nepochs=",epochs," hidden_nodes=",hidden_nodes," learning_rate=",learning_rate/100)
#nn1
            epochs1 = epochs
            hidden_nodes1 = hidden_nodes
            learning_rate1 = learning_rate/100
            
            #nn2
            epochs2 = epochs
            hidden_nodes2 = hidden_nodes
            learning_rate2 = learning_rate/100
            
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(X_tr[0].shape[0], hidden_nodes1, bias=False), 
                torch.nn.BatchNorm1d(hidden_nodes1),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_nodes1, 10),
                torch.nn.Softmax(dim=1)
            )
            
            nn2 = torch.nn.Sequential(
                torch.nn.Linear(X_tr[0].shape[0], hidden_nodes2, bias=True), 
                torch.nn.BatchNorm1d(hidden_nodes2),
                torch.nn.ReLU(),
            #    torch.nn.Linear(hidden_nodes2, hidden_nodes2),
             #   torch.nn.ReLU(),
                torch.nn.Linear(hidden_nodes2, 10), 
                torch.nn.Softmax(dim=1)
            )
            
#            print(nn1)
#            print(nn2)



            criterion = torch.nn.CrossEntropyLoss()
            optimizer1 = torch.optim.Rprop(nn1.parameters())
            optimizer2 = torch.optim.SGD(nn2.parameters(), lr=learning_rate2)
            
            loss_values1 = []
            loss_values2 = []
            
            for t in range(epochs1):
                optimizer1.zero_grad()
                output1 = nn1.forward(X_train)
                y_train=y_train.long()
                loss1 = criterion(output1, y_train)
                loss1.backward()
                optimizer1.step()
            
                loss_values1.append(loss1.item() / len(X_train))
                
            for t in range(epochs2):
                optimizer2.zero_grad()
                output2 = nn2.forward(X_train)
                loss2 = criterion(output2, y_train)
                loss2.backward()
                optimizer2.step()
            
                loss_values2.append(loss2.item() / len(X_train))
            
#            print(loss1.item())
#            print(loss2.item())
#            
#            if (loss1.item()<minl):
#                minl=loss1.item()
#                minepochs=epochs1
#                minnodes=hidden_nodes1
#                minrate=learning_rate1
#                IsBiased="False"
#               
#            if (loss2.item()<minl):
#                minl=loss1.item()
#                minepochs=epochs1
#                minnodes=hidden_nodes2
#                minrate=learning_rate2
#                IsBiased="True"
                

            # predicted classes for the training set - NN1
            y_pred_tr_nn1 = nn1.forward(X_train).data.numpy().argmax(axis=1)
            
            # error for training set - NN1
            train_error_nn1 = sklearn.metrics.zero_one_loss(y_train, y_pred_tr_nn1)
            
            # predicted classes for the test set - NN1
            y_pred_nn1 = nn1.forward(X_test).data.numpy().argmax(axis=1)
            
            # error for test set - NN1
            test_error_nn1 = sklearn.metrics.zero_one_loss(y_test, y_pred_nn1)
            
            # plot confusion matrix
#            plt.matshow(sklearn.metrics.confusion_matrix(y_test, y_pred_nn1))
#            plt.title("NN1: training error = %.2f, test error = %.2f" % (train_error_nn1, test_error_nn1))
#            plt.show()
            
            #--------------------------------------------------------------------------------------------------------------------
            
            # predicted classes for the training set - NN2
            y_pred_tr_nn2 = nn2.forward(X_train).data.numpy().argmax(axis=1)
            
            # error for training set - NN2
            train_error_nn2 = sklearn.metrics.zero_one_loss(y_train, y_pred_tr_nn2)
            
            # predicted classes for the test set - NN1
            y_pred_nn2 = nn2.forward(X_test).data.numpy().argmax(axis=1)
            
            # error for test set - NN2
            test_error_nn2 = sklearn.metrics.zero_one_loss(y_test, y_pred_nn2)
            
            # plot confusion matrix
#            plt.matshow(sklearn.metrics.confusion_matrix(y_test, y_pred_nn2))
#            plt.title("NN2: training error = %.2f, test error = %.2f" % (train_error_nn2, test_error_nn2))
#            plt.show()
#
#
#            plt.title("NN1 and NN2 - Losses over number of epochs")        
#            plt.plot(loss_values1[0:], label='NN1')
#            plt.plot(loss_values2[0:], label='NN2')
#            plt.xlabel('number of epochs')
#            plt.ylabel('loss')
#            plt.legend(loc='upper right')
#            plt.show()
#            
            if (train_error_nn1<mintrain):
                mintrain=train_error_nn1
                minepochs_train=epochs1
                minnodes_train=hidden_nodes
                minrate_train=learning_rate
                IsBiased_train="False"

            if (train_error_nn2<mintrain):
                mintrain=train_error_nn2
                minepochs_train=epochs1
                minnodes_train=hidden_nodes
                minrate_train=learning_rate
                IsBiased_train="True"
                
            if (test_error_nn1<mintest):
                mintest=test_error_nn1
                minepochs_test=epochs1
                minnodes_test=hidden_nodes
                minrate_test=learning_rate
                IsBiased_test="False"

            if (test_error_nn2<mintest):
                mintest=test_error_nn2
                minepochs_test=epochs1
                minnodes_test=hidden_nodes
                minrate_test=learning_rate
                IsBiased_test="True"
print("\n==========Results for Hasy dataset===========\n")             
print("\nFor TestSet min error=",mintest," for ecpochs=",minepochs_test," nodes=",minnodes_test," rate=",minrate_test/100, " biased=", IsBiased_test,"\n") 
print("\nFor TrainSet min error=",mintrain," for ecpochs=",minepochs_train," nodes=",minnodes_train," rate=",minrate_train/100, " biased=", IsBiased_train,"\n") 
from sklearn.ensemble import RandomForestClassifier

print("\n==========RF Classifier for Hasy dataset===========\n")
rf = RandomForestClassifier(max_depth=15, random_state=0, n_estimators=200)
rf.fit(X_train, y_train)
y_predtrain_rf = rf.predict(X_train)
train_error_rf = sklearn.metrics.zero_one_loss(y_train, y_predtrain_rf)
y_pred_rf = rf.predict(X_test)
test_error_rf = sklearn.metrics.zero_one_loss(y_test, y_pred_rf)
print("\nFor TestSet error=",test_error_rf)
print("\nFor TrainSet error=",train_error_rf)

# plot confusion matrix
#plt.matshow(sklearn.metrics.confusion_matrix(y_test, y_pred_rf))
#plt.title("RFC: training error = %.2f, test error = %.2f" % (train_error_rf, test_error_rf))
#plt.show()

print("\n==========Iterating for SLD dataset===========\n")
X = np.load('X.npy')
y = np.load('Y.npy').argmax(axis=1)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=0.8, test_size=0.2)
y_test_var = np.var(y_te)

X_tr = X_tr.reshape(X_tr.shape[0],X_tr.shape[1]*X_tr.shape[2])
X_te = X_te.reshape(X_te.shape[0],X_te.shape[1]*X_te.shape[2])

X_train = torch.from_numpy(np.array(X_tr, dtype='float32'))
y_train = torch.from_numpy(np.array(y_tr, dtype='int_'))
X_test = torch.from_numpy(np.array(X_te, dtype='float32'))
y_test = torch.from_numpy(np.array(y_te, dtype='int_'))


mintrain=10000
mintest=10000

for epochs in range(20,121,20):
    for hidden_nodes in range(5,26,5):
        for learning_rate in range(1,10,2):
            print("\nepochs=",epochs," hidden_nodes=",hidden_nodes," learning_rate=",learning_rate/100)
#nn1
            epochs1 = epochs
            hidden_nodes1 = hidden_nodes
            learning_rate1 = learning_rate/100
            
            #nn2
            epochs2 = epochs
            hidden_nodes2 = hidden_nodes
            learning_rate2 = learning_rate/100
            
            nn1 = torch.nn.Sequential(
                torch.nn.Linear(X_tr[0].shape[0], hidden_nodes1, bias=False), 
                torch.nn.BatchNorm1d(hidden_nodes1),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_nodes1, 10),
                torch.nn.Softmax(dim=1)
            )
            
            nn2 = torch.nn.Sequential(
                torch.nn.Linear(X_tr[0].shape[0], hidden_nodes2, bias=True), 
                torch.nn.BatchNorm1d(hidden_nodes2),
                torch.nn.ReLU(),
            #    torch.nn.Linear(hidden_nodes2, hidden_nodes2),
             #   torch.nn.ReLU(),
                torch.nn.Linear(hidden_nodes2, 10), 
                torch.nn.Softmax(dim=1)
            )
            
#            print(nn1)
#            print(nn2)



            criterion = torch.nn.CrossEntropyLoss()
            optimizer1 = torch.optim.Rprop(nn1.parameters())
            optimizer2 = torch.optim.SGD(nn2.parameters(), lr=learning_rate2)
            
            loss_values1 = []
            loss_values2 = []
            
            for t in range(epochs1):
                optimizer1.zero_grad()
                output1 = nn1.forward(X_train)
                y_train=y_train.long()
                loss1 = criterion(output1, y_train)
                loss1.backward()
                optimizer1.step()
            
                loss_values1.append(loss1.item() / len(X_train))
                
            for t in range(epochs2):
                optimizer2.zero_grad()
                output2 = nn2.forward(X_train)
                loss2 = criterion(output2, y_train)
                loss2.backward()
                optimizer2.step()
            
                loss_values2.append(loss2.item() / len(X_train))
            
#            print(loss1.item())
#            print(loss2.item())
#            
#            if (loss1.item()<minl):
#                minl=loss1.item()
#                minepochs=epochs1
#                minnodes=hidden_nodes1
#                minrate=learning_rate1
#                IsBiased="False"
#               
#            if (loss2.item()<minl):
#                minl=loss1.item()
#                minepochs=epochs1
#                minnodes=hidden_nodes2
#                minrate=learning_rate2
#                IsBiased="True"
                

            # predicted classes for the training set - NN1
            y_pred_tr_nn1 = nn1.forward(X_train).data.numpy().argmax(axis=1)
            
            # error for training set - NN1
            train_error_nn1 = sklearn.metrics.zero_one_loss(y_train, y_pred_tr_nn1)
            
            # predicted classes for the test set - NN1
            y_pred_nn1 = nn1.forward(X_test).data.numpy().argmax(axis=1)
            
            # error for test set - NN1
            test_error_nn1 = sklearn.metrics.zero_one_loss(y_test, y_pred_nn1)
            
            # plot confusion matrix
#            plt.matshow(sklearn.metrics.confusion_matrix(y_test, y_pred_nn1))
#            plt.title("NN1: training error = %.2f, test error = %.2f" % (train_error_nn1, test_error_nn1))
#            plt.show()
            
            #--------------------------------------------------------------------------------------------------------------------
            
            # predicted classes for the training set - NN2
            y_pred_tr_nn2 = nn2.forward(X_train).data.numpy().argmax(axis=1)
            
            # error for training set - NN2
            train_error_nn2 = sklearn.metrics.zero_one_loss(y_train, y_pred_tr_nn2)
            
            # predicted classes for the test set - NN1
            y_pred_nn2 = nn2.forward(X_test).data.numpy().argmax(axis=1)
            
            # error for test set - NN2
            test_error_nn2 = sklearn.metrics.zero_one_loss(y_test, y_pred_nn2)
            
            # plot confusion matrix
#            plt.matshow(sklearn.metrics.confusion_matrix(y_test, y_pred_nn2))
#            plt.title("NN2: training error = %.2f, test error = %.2f" % (train_error_nn2, test_error_nn2))
#            plt.show()
#
#
#            plt.title("NN1 and NN2 - Losses over number of epochs")        
#            plt.plot(loss_values1[0:], label='NN1')
#            plt.plot(loss_values2[0:], label='NN2')
#            plt.xlabel('number of epochs')
#            plt.ylabel('loss')
#            plt.legend(loc='upper right')
#            plt.show()
#            
            if (train_error_nn1<mintrain):
                mintrain=train_error_nn1
                minepochs_train=epochs1
                minnodes_train=hidden_nodes
                minrate_train=learning_rate
                IsBiased_train="False"

            if (train_error_nn2<mintrain):
                mintrain=train_error_nn2
                minepochs_train=epochs1
                minnodes_train=hidden_nodes
                minrate_train=learning_rate
                IsBiased_train="True"
                
            if (test_error_nn1<mintest):
                mintest=test_error_nn1
                minepochs_test=epochs1
                minnodes_test=hidden_nodes
                minrate_test=learning_rate
                IsBiased_test="False"

            if (test_error_nn2<mintest):
                mintest=test_error_nn2
                minepochs_test=epochs1
                minnodes_test=hidden_nodes
                minrate_test=learning_rate
                IsBiased_test="True"
print("\n==========Results for SLD dataset===========\n")
print("\nFor TestSet min error=",mintest," for ecpochs=",minepochs_test," nodes=",minnodes_test," rate=",minrate_test/100, " biased=", IsBiased_test,"\n") 
print("\nFor TrainSet min error=",mintrain," for ecpochs=",minepochs_train," nodes=",minnodes_train," rate=",minrate_train/100, " biased=", IsBiased_train,"\n")
from sklearn.ensemble import RandomForestClassifier

print("\n==========RF Classifier for SLD dataset===========\n")
rf = RandomForestClassifier(max_depth=15, random_state=0, n_estimators=200)
rf.fit(X_train, y_train)
y_predtrain_rf = rf.predict(X_train)
train_error_rf = sklearn.metrics.zero_one_loss(y_train, y_predtrain_rf)
y_pred_rf = rf.predict(X_test)
test_error_rf = sklearn.metrics.zero_one_loss(y_test, y_pred_rf)
print("\nFor TestSet error=",test_error_rf)
print("\nFor TrainSet error=",train_error_rf)


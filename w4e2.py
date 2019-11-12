# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 05:59:28 2019

@author: Paul
"""

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.ensemble import RandomForestClassifier
from tpot import TPOTClassifier
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df=pd.read_csv("hasy-data-labels.csv")
print(df)
df2=df.loc[(df['symbol_id'] >= 70) & (df['symbol_id'] <= 80)]
print(df2)

total_rows = df2.count
print (total_rows)

df3 = pd.DataFrame({'Data': []})
lab_df = pd.DataFrame({'Labels': []})
sid_df = pd.DataFrame({'ID': []})
path_df =pd.DataFrame({'path': []})


for i,j in df2.iterrows():
    print(j)
    path = df.at[i,'path']
    img = Image.open(path).convert('L')
    sid = df.at[i,'symbol_id']
    
    lab_df = lab_df.append({'Labels': img}, ignore_index=True)
    sid_df = sid_df.append({'ID': sid}, ignore_index=True)
    path_df = path_df.append({'path': path}, ignore_index=True) 
    
    arr = np.array(img)  
    flat_arr = arr.ravel()
    df3 = df3.append({'Data': flat_arr}, ignore_index=True)


df = pd.concat([df3,sid_df], axis=1)
df = pd.concat([df,path_df], axis=1)





from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2)

print(train)

Xtrain=train[["Data"]]
Xtrain = pd.DataFrame(Xtrain.Data.values.tolist(), index= Xtrain.index)
ytrain=train[["ID"]]

Xtest=test[["Data"]]
Xtest = pd.DataFrame(Xtest.Data.values.tolist(), index= Xtest.index)
ytest=test[["ID"]]

clf=RandomForestClassifier()

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(Xtrain,ytrain)

y_pred=clf.predict(Xtest)
ytestvalues=ytest.values
s=0
for i in range(0,len(y_pred)):
    if (ytestvalues[i]==y_pred[i]):
        s=s+1
success_rate=s/len(y_pred)*100    
print("\n\n ACCURACY WITHOUT TUNING IS ", success_rate,"%") 

acc=np.zeros(10)
n_of_est=np.zeros(10)

for i in range(1,11):
        size=i*20
        train, test = train_test_split(df, test_size=0.2)

#        print(train)
        
        Xtrain=train[["Data"]]
        Xtrain = pd.DataFrame(Xtrain.Data.values.tolist(), index= Xtrain.index)
        ytrain=train[["ID"]]
        
        Xtest=test[["Data"]]
        Xtest = pd.DataFrame(Xtest.Data.values.tolist(), index= Xtest.index)
        ytest=test[["ID"]]
        
        clf=RandomForestClassifier(n_estimators=size)
        
        #Train the model using the training sets y_pred=clf.predict(X_test)
        clf.fit(Xtrain,ytrain)
        
        y_pred=clf.predict(Xtest)
        ytestvalues=ytest.values
        s=0
        for j in range(0,len(y_pred)):
            if (ytestvalues[j]==y_pred[j]):
                s=s+1
        success_rate=s/len(y_pred)*100    
        print("\n\n ACCURACY WITH ",size, " estimators is ",success_rate ,"%") 

        acc[i-1]=success_rate
        n_of_est[i-1]=i*20
        
plt.plot(acc,  n_of_est)
plt.suptitle('No of estimators VS %accuracy')
plt.show()

maxacc=-1


for i in range(1,11):
    for m in range (1,11):
        size=i*20
        depth=m*5

        train, test1 = train_test_split(df, test_size=0.2)
        valid, test = train_test_split(test1, test_size=0.5)
#        print(train)
        
        Xtrain=train[["Data"]]
        Xtrain = pd.DataFrame(Xtrain.Data.values.tolist(), index= Xtrain.index)
        ytrain=train[["ID"]]
        
        Xvalid=valid[["Data"]]
        Xvalid = pd.DataFrame(Xvalid.Data.values.tolist(), index= Xvalid.index)
        yvalid=valid[["ID"]]
        
        Xtest=test[["Data"]]
        Xtest = pd.DataFrame(Xtest.Data.values.tolist(), index= Xtest.index)
        ytest=test[["ID"]]
    
        
        
        clf=RandomForestClassifier(n_estimators=size, max_depth=depth)
        
        #Train the model using the training sets y_pred=clf.predict(X_test)
        clf.fit(Xtrain,ytrain)
        
        y_pred=clf.predict(Xvalid)
        yvalidvalues=yvalid.values
        s=0
        for j in range(0,len(y_pred)):
            if (yvalidvalues[j]==y_pred[j]):
                s=s+1
        success_rate=s/len(y_pred)*100    
        print("\n\n ACCURACY (validator) WITH ",size, " estimators and max depth=", depth," is ",success_rate ,"%") 
        
     
        
        
        if success_rate>maxacc: 
            maxacc=success_rate
            est=size
            bestdepth=depth
            Xtesttokeep=Xtest
            ytesttokeep=ytest
            Xtraintokeep=Xtrain
            ytraintokeep=ytrain
        
        acc[i-1]=success_rate
        n_of_est[i-1]=i*20
        
print("\n\n BEST ACCURACY SIZE ACCORDING TO VALIDATOR ",est, "AND IDEAL MAX_DEPTH=",bestdepth)

clf=RandomForestClassifier(n_estimators=est, max_depth=bestdepth)
clf.fit(Xtraintokeep,ytraintokeep)
y_pred=clf.predict(Xtesttokeep)
ytesttokeep=ytesttokeep.values
s=0
for j in range(0,len(y_pred)):
    if (ytesttokeep[j]==y_pred[j]):
        s=s+1
    success_rate2=s/len(y_pred)*100    
print("\n\n ACCURACY (test data) WITH ",est, " estimators and max_depth=",bestdepth, " (chosen from validator) is ",success_rate2 ,"%") 

#plt.plot(acc,  n_of_est)
#plt.suptitle('No of estimators VS %accuracy (validator data)')
#plt.show()

tpot = TPOTClassifier(generations=5, population_size=10, verbosity=2)
tpot.fit(Xtraintokeep, ytraintokeep)
print(tpot.score(Xtesttokeep, ytesttokeep))
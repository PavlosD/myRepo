# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 17:59:15 2019

@author: Paul
"""

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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



from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2)

print(train)

Xtrain=train[["Data"]]
Xtrain = pd.DataFrame(Xtrain.Data.values.tolist(), index= Xtrain.index)
ytrain=train[["ID"]]

Xtest=test[["Data"]]
Xtest = pd.DataFrame(Xtest.Data.values.tolist(), index= Xtest.index)
ytest=test[["ID"]]



from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

items_counts = train['ID'].value_counts()
max_item = items_counts.idxmax()
print ("max count of", max_item)

for K in range(0,10):
    #LOGISTIC REGRESSION
    print("\n =================  ITERATING NOW FOR ===================", K)
    y1 = pd.DataFrame({'Is70': []})
    y1['Is70'] = (ytrain['ID'] == 70+K) 
    ytest1 = pd.DataFrame({'Is70': []})
    ytest1['Is70'] = (ytest['ID'] == 70+K) 
    
    model.fit(Xtrain, y1) 
    
    y_pred=model.predict(Xtest)
    ytest1values=ytest1.values
    
    s=0
    f=0
    for i in range(0,len(y_pred)):
        if (ytest1values[i]==y_pred[i]):
            s=s+1
        else:
            if (y_pred[i]):
                print("FALSE POSITIVE")
            else:
                print("FALSE NEGATIVE")
            f=f+1
            path2=test.iat[i,2]
            img=mpimg.imread(path2)
            imgplot = plt.imshow(img)
            plt.show()
            
    success_rate=s/(s+f)*100    
    print("ACCURACY FOR", K , " IS ", success_rate,"%") 
    
guess=max_item

s=0
f=0

ytestvalues = ytest.values
for i in range(0,len(ytestvalues)):
    if (ytestvalues[i]==guess):
            s=s+1
    else:
            f=f+1
    
success_rate=s/(s+f)*100    
print("ACCURACY FOR GUESSES IS ", success_rate,"%") 
        

    
    
    

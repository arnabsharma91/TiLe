
# coding: utf-8

# In[1]:


import pandas as pd
import csv as cv
#K-NN classifier
#Import Library
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import numpy as np
from sklearn.naive_bayes import BaseNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import fileinput


# In[ ]:


inc = 0


#Function to get the input data from configuration file
predicted1 = []
fp = open("Config.txt")
InputData = ""
for i, line in enumerate(fp):
    if i == 0:
       InputData = fp.readline()
       #print(InputData)
       break
fp.close()


with open('Inpdata.txt') as f:
    count = int(f.readline())

    
if(count == 0):
    #training the model
    df = pd.read_csv('./FairData/' + InputData)
    data = df.as_matrix()
    training, test = data[:10,:], data[10:,:]
    X = training[:, :-1]
    Y = training[:, -1]
    
    model = GaussianNB()
    model.fit(X, Y)
    
    #testing phase
    #sf = pd.read_csv('./FairData/TestOldData.csv')
    #testData = sf.as_matrix()
    x_test = test[:, :-1] 
    predictBefore = model.predict(x_test)
    predictWrite = np.zeros((predictBefore.shape[0], 1))
    for i in range(predictBefore.shape[0]):   
        predictWrite[i] = predictBefore[i]
    with open('Prediction.csv', 'w') as csvfile:
        fieldnames = ['pred1']
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(predictWrite)
    print('Prediction before applying the functions:')
    print(predictBefore)
else:
    df = pd.read_csv('./FairData/ModifiedData.csv')
    dataAfter = df.as_matrix()
    training, test = dataAfter[:10,:], dataAfter[10:,:]
    X = training[:, :-1]
    Y = training[:, -1]
    
    #df = pd.read_csv('./FairData/' + InputData)
    #D = np.array(df.values, dtype=None)
    #np.random.shuffle(D)
    #X = D[:, :-1]
    #Y = D[:, -1]
    #np.random.shuffle(np.transpose(X))
    #training, test = D[:10,:], D[10:,:]
    
    model = GaussianNB()
    model.fit(X, Y)
    
    #sf = pd.read_csv('./FairData/TestOldData.csv')
    #testData = sf.as_matrix()
    x_test = test[:, :-1]
    predictAfter = model.predict(x_test)

# Train the model using the training sets and check score
    
    df1 = pd.read_csv('Prediction.csv')
    Temp = df1.as_matrix()
    print('Prediction after applying the functions:')
    print(predictAfter)
    for i in range(predictAfter.shape[0]):        
        if(Temp[i] == predictAfter[i]):
            inc = inc +1
    inc = (inc /predictAfter.shape[0])*100
    print(inc)




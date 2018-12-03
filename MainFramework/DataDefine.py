
# coding: utf-8

# In[2]:


import pandas as pd
import csv as cv
from sklearn import tree
import numpy as np
import fileinput
import TrainData
from sklearn.feature_selection import VarianceThreshold
import math
from random import randint


# In[6]:


def funcDataImmuno(model, par):
    dfAgain = pd.read_csv('ModifiedDataBase.csv')
    #print(dfAgain)
    dataAfter = dfAgain.as_matrix()
    dataAfter = np.round(dataAfter, 2)
    training, test = dataAfter[:par,:], dataAfter[par:,:]
    modelAfter = model
    measure = TrainData.funcTrainTestAfter(training, test, modelAfter)
    return measure
    


# In[8]:


def funcDataGerman(model):
    dfAgain = pd.read_csv('ModifiedDataBase.csv')
    dfTest = pd.read_csv('GermanCreditTestDataNew.csv')
    modelAfter = model
    measure = TrainData.funcTrainTestAfter(dfAgain, modelAfter, dfTest)
    return measure
    


# In[10]:



    


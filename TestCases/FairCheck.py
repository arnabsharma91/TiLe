
# coding: utf-8

# In[1]:


#import part

import numpy as np
import scipy as sp
import arff as arff
import csv as cv
import pandas as pd
import random as rd
from scipy import io
import time
from scipy.stats import kurtosis
from sklearn.feature_selection import VarianceThreshold
import math
import DataDefine
from random import randint
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[8]:


#Function to flip the batch of the data
def funcPermRowsBat(df):
    
    dfTrain = df
    dfNew = pd.DataFrame()
    dfAgain = pd.DataFrame()
    row = df.shape[0]
    i = df.shape[0] -1

    batSize = round(row/10)
    #print(batSize)
    while(i >= 0):
        data = dfTrain.iloc[[i]]
        dfNew = dfNew.append(data)
        
        if((i % batSize) == 0):
            Reversed_df = dfNew.iloc[::-1]
            #print(Reversed_df)
            dfAgain = dfAgain.append(Reversed_df)
            dfNew.drop(dfNew.index, inplace= True)
        i= i-1
    dfAgainNew = dfAgain        
    dfAgainNew.to_csv('ModifiedDataBase.csv', index= False, header=True)


# In[9]:


#Shuffling feature names
def funcShuffleFeatures(df):
    noOfAttributes = df.shape[1] -1
    noOfAttributesHalf = round(noOfAttributes/2) -1
    for i in range(0, noOfAttributesHalf):
        temp = df.columns.values[i]
        df.columns.values[i] = df.columns.values[noOfAttributesHalf-i]
        df.columns.values[noOfAttributesHalf-i] = temp                  
    df.to_csv('ModifiedDataBase.csv', index = False, header = True)


# In[3]:


#Function to order the rows in ascending order
def funcPermuteRowsAsc(df):
    dfTrainMod=df.sort_values('Class')
    dfNew = dfTrainMod
    dfNew.to_csv('ModifiedDataBase.csv', index = False, header = True)
    


# In[ ]:


#Function to order the rows in descending order
def funcPermuteRowsDesc(df):
    dfTrainMod=df.sort_values('Class', ascending=False)
    dfNew = dfTrainMod
    dfNew.to_csv('ModifiedDataBase.csv', index = False, header = True)
    


# In[ ]:


#function to organize the data in class value alternating order
def funcPermuteRowsAlt(df):
    dfTrainMod=df.sort_values('Class')
    dfTrainMod.to_csv('ModifiedDataBaseTest.csv', index = False, header = True)
    df = pd.read_csv('ModifiedDataBaseTest.csv')
    n=df.loc[df.Class == 0, 'Class'].count()
    end = df.shape[0]
    for i in range(0, n-1):
        if((i%2)==1):
            df1 = df.iloc[i]
            df2 = df.iloc[end-i]
            temp = df.iloc[i].copy()
            df.iloc[i] = df.iloc[end-i]
            df.iloc[end-i] = temp 
    
    dfNew = df    
    dfNew.to_csv('ModifiedDataBase.csv', index = False, header = True)


# In[ ]:



    


# In[ ]:


#Function to reverse the whole dataset
def funcPermuteRows(df):
    df.sort_index(inplace=True, ascending= False)
    df.to_csv('ModifiedDataBase.csv', index = False, header = True)


# In[1]:


#Function to randomly permute columns
def funcPermuteCols(df):
    df = df.sample(frac=1, axis=1).reset_index(drop=True)
    dfClass = df[['Class']]
    df.drop('Class', axis=1, inplace=True)
    dfNew = pd.concat([df, dfClass], axis=1)             
    dfNew.to_csv('ModifiedDataBase.csv', index = False, header = True)


# In[2]:


#Function to randomly permute rows
def funcRandPermuteRows(df):
    dfTrain = df.sample(frac=1, axis=0).reset_index(drop=True)
    dfTrain.to_csv('ModifiedDataBase.csv', index = False, header = True)


# In[4]:



         
        
        
        


# In[ ]:



    


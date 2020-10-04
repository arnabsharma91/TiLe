

import numpy as np
import scipy as sp
import csv as cv
import pandas as pd
import random as rd
from scipy import io
import time
import math
from random import randint


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


#Function to order the rows in ascending order
def funcPermuteRowsAsc(df):
    dfTrainMod=df.sort_values('Class')
    dfNew = dfTrainMod
    dfNew.to_csv('ModifiedDataBase.csv', index = False, header = True)
    


#Function to order the rows in descending order
def funcPermuteRowsDesc(df):
    dfTrainMod=df.sort_values('Class', ascending=False)
    dfNew = dfTrainMod
    dfNew.to_csv('ModifiedDataBase.csv', index = False, header = True)
    

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


#Function to reverse the whole dataset
def funcReverseRows(df):
    df.sort_index(inplace=True, ascending= False)
    df.to_csv('ModifiedDataBase.csv', index = False, header = True)

    
#Function to randomly permute columns
def funcPermuteCols(df):
    df = df.sample(frac=1, axis=1).reset_index(drop=True)
    dfClass = df[['Class']]
    df.drop('Class', axis=1, inplace=True)
    dfNew = pd.concat([df, dfClass], axis=1)             
    dfNew.to_csv('ModifiedDataBase.csv', index = False, header = True)


#Function to randomly permute rows
def funcRandPermuteRows(df):
    dfTrain = df.sample(frac=1, axis=0).reset_index(drop=True)
    dfTrain.to_csv('ModifiedDataBase.csv', index = False, header = True)




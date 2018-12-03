
# coding: utf-8

# In[2]:


import pandas as pd
import csv as cv
import numpy as np
import fileinput
import graphviz
import FairCheck 
import math
from sklearn.neural_network import MLPClassifier
import math
from random import randint
import random as rd
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import sys
import SVMEquiCheck
import RandForEquiCheck
import AdaBoostEquiCheck
import LogRegEquiCheck
import EquiTrain
import os
import scipy.stats as st


# In[1]:


def funcTrainTest(model):
 
 

 noOfRows = 0 
 temp = 0
 count = 0
 sim_measure = 0
 #Function to get the input data from configuration file
 
    
 
 train_ratio = float(input("Enter the percentage value for train ratio"))
 confLevel = float(input("Enter the confidence level in percentage"))   
 for count in range(0, 9):    
      #Real-world data 
     permRowsCount = 0
     permColumnCount = 0
     permShuffleCount = 0
     data = np.zeros((3, ))
     if(count > 0):
        os.remove('TestingData.csv')
        
        
     datafile = open('Config.txt', 'w') 
     if(count == 0):
         df = pd.read_csv('DataRepository/ImmunotherapyData.csv') 
         print("for ImmunoData")
         data = df.values
         datafile.write("DataRepository/ImmunotherapyData.csv")
     elif(count == 1):
         df = pd.read_csv('DataRepository/GermanCreditData.csv')
         print("for GermanCredit data")
         data = df.values
         datafile.write("DataRepository/GermanCreditData.csv")
     elif(count == 2):
         df = pd.read_csv('DataRepository/BreastCancerDataWisconsinOriginal.csv')
         print("for BreastCancer data")
         data = df.values
         datafile.write("DataRepository/BreastCancerDataWisconsinOriginal.csv")
     elif(count == 3):
         df = pd.read_csv('DataRepository/LungCancerData.csv')
         print("for Lung Cancer data")
         data = df.values
         datafile.write("DataRepository/LungCancerData.csv") 
     elif(count == 4):
         df = pd.read_csv('DataRepository/CrimeData.csv')
         print("for crime data")
         data = df.values 
         datafile.write("DataRepository/CrimeData.csv")   
     elif(count == 5):
         df = pd.read_csv('DataRepository/SEData.csv')
         print("for Software engineering data")
         data = df.values
         datafile.write("DataRepository/SEData.csv")
     elif(count == 6):
         df = pd.read_csv('DataRepository/AdultData.csv')
         print("for Adult data")
         data = df.values
         datafile.write("DataRepository/AdultData.csv")    
     elif(count == 7):
         df = pd.read_csv('DataRepository/OccupancyData.csv')
         print("for Occupancy data")
         data = df.values
         datafile.write("DataRepository/OccupancyData.csv")  
            
     elif(count == 8):
         df = pd.read_csv('DataRepository/VoiceData.csv')
         print("for Voice data")
         data = df.values
         datafile.write("DataRepository/VoiceData.csv") 
    
     datafile.close()   
     
     flag = False
     
    
     #Dealing with round-off error   
     data = np.round(data, 2)     
     
     training = data  
    
     #Test data taken from the training data
     noRow = df.shape[0]   
     test_param = round((noRow*train_ratio)/100) 
     #print(test_param)   
     generateTest(test_param, data, df)   
     
     #Test data from Mean, Median, Mode, Min, Max values of each feature
     generateMeMo(df) 
    
     #Test data randomly generated with a user given confidence level and error 0.01
     test = generateRanTest(df, confLevel, data) 
     test = np.round(test, 2)
    
     #Training the model before applying the functions
     modelBefore = model
     filename = sys.argv[-1]   
     #print(filename)
     #flag = funcCheckModel(modelBefore, filename, training, rowNo, df, count)   
     #print(flag)
     #if(flag==True):
      #   SVMTest.funcTrainCheck(model) 
        
     if(flag==False):   
         #follow normal path
         funcTrainTestBefore(training, test, modelBefore)
         
         with open('Config.txt') as f:
            InpDataFile = f.readline()
         f.close()
         
         temp = 0   
          
         dfS = pd.read_csv(InpDataFile)
         print("Average similarity after shuffling feature names")
         FairCheck.funcShuffleFeatures(dfS)
         temp = funcTrainTestAfter(model, test)  
         print("Deviation after applying function") 
         print(100-temp)
         permShuffleCount = 100-temp
         print('\n') 
         temp = 0   
        
        
         
          #permutation of columns
         print("Average similarity after column permutation")
         dfC = pd.read_csv(InpDataFile)
         #noOfColumnsPerm = round((math.factorial(noOfColumns))/5) 
         noOfColumnsPerm = 1
    
         for i in range(0, noOfColumnsPerm):
              FairCheck.funcPermuteCols(dfC)
              temp = temp + funcTrainTestAfter(model, test)  
         temp = temp/noOfColumnsPerm    
         #print(diffMeasureCol)
         print("Deviation after applying function")
         permColumnCount = 100-temp   
         print(100-temp)
         print('\n')   
         temp = 0
         
     
         dfR1 = pd.read_csv(InpDataFile)
         #Permutation of rows systematically
         print("Average similarity after reversing the whole dataset")
         
         FairCheck.funcPermuteRows(dfR1)
         temp = funcTrainTestAfter(model, test)  
         permRowsCount = permRowsCount + (100-temp)    
         #print(diffMeasureRow)
         print("Deviation after applying function") 
         print(100-temp)
         print('\n')   
        
     
         temp = 0
     
         dfR2 = pd.read_csv(InpDataFile)
         #Permutation of rows 
         print("Average similarity after ascending class row organizing")
         FairCheck.funcPermuteRowsAsc(dfR2)
         temp = funcTrainTestAfter(model, test)  
         permRowsCount = permRowsCount + (100-temp)   
         print("Deviation after applying function") 
         print(100-temp)
         print('\n')   
        
     
         temp = 0
        
     
        #Permutation of rows 
         dfR3 = pd.read_csv(InpDataFile)
         print("Average similarity after altermative class row organizing")
         FairCheck.funcPermuteRowsAlt(dfR3)
         temp = funcTrainTestAfter(model, test)
         permRowsCount = permRowsCount + (100-temp)    
         print("Deviation after applying function") 
         print(100-temp)
         print('\n')   
        
     
         temp = 0
        
         #Permutation of rows 
         dfR4 = pd.read_csv(InpDataFile)   
         print("Average similarity after descending class row organizing")
         FairCheck.funcPermuteRowsDesc(dfR4)
         temp = funcTrainTestAfter(model, test)
         permRowsCount = permRowsCount + (100-temp) 
         print("Deviation after applying function") 
         print(100-temp)
         print('\n')   
        
     
         temp = 0   
        
    
          #Random row permutations
         dfR5 = pd.read_csv(InpDataFile)
         print("Average similarity after Row random permutation")   
         #print("After Row random permutation")   
         #noOfRowsPerm = round((math.factorial(noOfRows))/5)
         noOfRowsPerm = 1
            
         #print(noOfRowsPerm)   
         for i in range(0, noOfRowsPerm):
              FairCheck.funcRandPermuteRows(dfR5)
              temp = temp + funcTrainTestAfter(model, test)  
         temp = temp/noOfRowsPerm  
         permRowsCount = permRowsCount + (100-temp)   
         #print(diffMeasureCol)
         print("Deviation after applying function")
         print(100-temp)
         print('\n')   
         temp = 0
        
         
          #Permutation of rows 
         dfR6 = pd.read_csv(InpDataFile)   
         print("Average similarity after batch shuffling")
         FairCheck.funcPermRowsBat(dfR6)
         temp = funcTrainTestAfter(model, test)
         permRowsCount = permRowsCount + (100-temp) 
         print("Deviation after applying function") 
         print(100-temp)
         print('\n')     
            
    
    
         
         
         permRowsCountAvg = permRowsCount/6
         if(count == 0):
             with open('PermResult.csv', 'w', newline = '') as csvfile:
                fieldnames = ['RowPerm', 'ColumnPerm', 'FeShuffle']
                fieldname = [permRowsCountAvg, permColumnCount, permShuffleCount]
                writer = cv.writer(csvfile)
                writer.writerow(fieldnames)
                writer.writerow(fieldname)
         else:
             dfPerm = pd.read_csv('PermResult.csv')
             dfPerm2 = pd.DataFrame({"RowPerm": [permRowsCountAvg], "ColumnPerm": [permColumnCount], "FeShuffle": [permShuffleCount]})
             dfPerm = dfPerm.append(dfPerm2)
             dfPerm.to_csv('PermResult.csv', index = False)   
        
    
     noOfRows = 0
     noOfColumns = 0   
     #print('\n')
     #print('\n')
        
 if(flag == False):
     dfRead = pd.read_csv('PermResult.csv')
     permRowsFinalCount = dfRead['RowPerm'].sum()/9
     permColsFinalCount = dfRead['ColumnPerm'].sum()/9
     permShuffleFinalCount = dfRead['FeShuffle'].sum()/9
     print("Row permutation bi_t(learn)")
     print(permRowsFinalCount)
     print("Column permutation bi_t(learn)")
     print(permColsFinalCount)
     print("Feature Shuffling bi_t(learn)")
     print(permShuffleFinalCount)
 else:
     dfRAg = pd.read_csv('PermResultEqui.csv') 
     permRowsEquiFinalCount = dfRAg['RowPerm'].sum()/9
     permColsEquiFinalCount = dfRAg['ColumnPerm'].sum()/9
     permShuffleEquiFinalCount = dfRAg['FeShuffle'].sum()/9
     if(permRowsEquiFinalCount >=0.01):
            EquiTrain.funcTrainTest(model)
     elif(permColsEquiFinalCount >=0.01):
            EquiTrain.funcTrainTest(model)
     elif(permShuffleEquiFinalCount >=0.01):
            EquiTrain.funcTrainTest(model)
     else:
            print("Models are equivalent after applying mm transformation")
 


# In[ ]:


def generateRanTest(df, conf_level, dataTrain):
    
    noCol = df.shape[1]
    noRow = df.shape[0]
    
    alpha = 1- (conf_level/100)
    alphaMod = alpha/2
    #print(alphaMod)
    confidenceLevel = abs(st.norm.ppf(alphaMod))
    m = ((pow(confidenceLevel, 2) * pow(0.5, 2))/pow(0.01, 2))
    noOfSamples = int(round(m/(1+((m-1)/noRow))))
    
    
    MaxArray = np.zeros((1, noCol-1))
    MinArray = np.zeros((1, noCol-1))
    temp = np.zeros((noCol-1, ))
    tempList = np.zeros((1, noCol-1))
    flg = False
    countSamples = 0
    for i in range(0, noCol-1):
        MaxArray[0][i] = df.iloc[:, i].max()
        MinArray[0][i] = df.iloc[:, i].min()
        
    while(countSamples <= noOfSamples):
        for i in range(0, noCol-1):
           ranInd = rd.randint(0, noRow-1) 
           tempList[0][i] = dataTrain[ranInd][i]
        for i in range(0, noCol-1):
           temp[i] = tempList[0][i] 
        if(countSamples > 0):
            dfT = pd.read_csv('TestingData.csv')                     
            data = dfT.values
            data = np.array(data)
        
            if(np.any(data == temp).all(axis=0)):
                break
            #print('ello')
            else:
            #print('world')
                with open('TestingData.csv', 'a', newline='') as csvfile:
                    writer = cv.writer(csvfile)
                    writer.writerows(tempList)
                countSamples = countSamples +1
                
        if(countSamples == 0):    
            with open('TestingData.csv', 'a', newline='') as csvfile:
                writer = cv.writer(csvfile)
                writer.writerows(tempList)
            countSamples = countSamples +1            
    
    dfTest = pd.read_csv('TestingData.csv')
    dataTest = dfTest.values   
    return dataTest    
                
    #dfAgain = pd.read_csv('TestingData.csv')
    #dfAgain.duplicated()


# In[ ]:


#Function to generate test data as mean, median, min, max values of each feature
def generateMeMo(dfT):
    n = dfT.shape[1] 
    MaxArray = np.zeros((1, n-1))
    MinArray = np.zeros((1, n-1))
    MeanArray = np.zeros((1, n-1))
    MedianArray = np.zeros((1, n-1))
    
    for i in range(0, n-1):
        MaxArray[0][i] = dfT.iloc[:, i].max()
        MinArray[0][i] = dfT.iloc[:, i].min()
        MeanArray[0][i] = dfT[dfT.columns[i]].mean()
        MedianArray[0][i] = dfT[dfT.columns[i]].median()   
        
        
    with open('TestingData.csv', 'a', newline = '') as csvfile:
        writer = cv.writer(csvfile)
        writer.writerows(MeanArray)
        writer.writerows(MedianArray)
        writer.writerows(MaxArray)
        writer.writerows(MinArray)  
        
        
    


# In[ ]:


#Function to generate test data from training data
def generateTest(tst_pm, data, df):
    
     testMatrix = np.zeros(((tst_pm+1), df.shape[1]))   
     flg = True
     testCount = 0   
     ratioTrack = []
     noOfRows = df.shape[0]
     #Choosing 
     while(testCount <= tst_pm):
        
        ratio = rd.randint(0, noOfRows-1)
            
        if(testCount >= 1):    
            flg = binSearch(ratioTrack, ratio)
            #print(ratioTrack)
            #print(ratio)
            #print(flg)
            if(flg == False):
                #print('world')
                ratioTrack.append(ratio)    
                testMatrix[testCount] = data[ratio]
                testCount = testCount +1
        if(testCount == 0):
            #print('hello')
            ratioTrack.append(ratio)     
            testMatrix[testCount] = data[ratio]
            testCount = testCount +1      
    
     #print(ratioTrack) 
     #print(data)      
     with open('TestingData.csv', 'w', newline='') as csvfile:
        fieldnames = df.columns.values
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(testMatrix)
     
     


# In[ ]:


#function to search for duplicate test data
def binSearch(alist, item):
    if len(alist) == 0:
        return False
    else:
        midpoint = len(alist)//2
        if alist[midpoint]==item:
          return True
        else:
          if item<alist[midpoint]:
           return binSearch(alist[:midpoint],item)
          else:
           return binSearch(alist[midpoint+1:],item)


# In[5]:


def funcTrainTestBefore(training, test, model):
    
    X = training[:, :-1]
    Y = training[:, -1]
    model = model.fit(X, Y)
                
    #print(model.coefs_)
                
                
    x_test = test[:, :-1]
    y_true = test[:, -1]
    
    
    predictBefore = model.predict(x_test)
    predictBefore = np.round(predictBefore, 1)
    
    
    
    predictWrite = np.zeros((predictBefore.shape[0], 1))
    for i in range(predictBefore.shape[0]):   
        predictWrite[i] = predictBefore[i]
    with open('Prediction.csv', 'w') as csvfile:
        fieldnames = ['pred1']
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(predictWrite)
    #print('Prediction before applying the functions:')
    #print(predictBefore)
    
    


# In[2]:



    


# In[1]:


def funcTrainTestAfter(model, test):
    inc = 0
    
    dfAgain = pd.read_csv('ModifiedDataBase.csv')
    training = dfAgain.values
    training = np.round(training, 2)
    
    X = training[:, :-1]
    Y = training[:, -1] 
    model = model.fit(X, Y)
    #print(model.coefs_)
    
    #dfTestAg = pd.read_csv('TestingData.csv')
    #test = dfTestAg.values
    
    x_test = test[:, :-1]
    y_true = test[:, -1]
    
    predictAfter = model.predict(x_test)
    predictAfter = np.round(predictAfter, 1)
    
   
    # Train the model using the training sets and check score

    df1 = pd.read_csv('Prediction.csv')
    Temp = df1.as_matrix()
    #print('Prediction after applying the functions:')
    #print(predictAfter)
    for i in range(predictAfter.shape[0]):        
        if(Temp[i] == predictAfter[i]):
            inc = inc +1
    inc = (inc /predictAfter.shape[0])*100
    #print(inc)
    return inc
    


# In[ ]:


def funcCheckModel(model, name, training, rowNo, df, cnt):
    if(("svm.LinearSVC(") in open (name).read()):
        SVMEquiCheck.funcModelCheck(model, training, rowNo, df, cnt)
        return True
    elif(("RandomForestClassifier(") in open (name).read()):
        RandForEquiCheck.funcModelCheck(model, training, rowNo, df, cnt)
        return True
    elif(("AdaBoostClassifier(") in open (name).read()):
        RandForEquiCheck.funcModelCheck(model, training, rowNo, df, cnt)
        return True
    elif(("LogisticRegression(") in open (name).read()):
        SVMEquiCheck.funcModelCheck(model, training, rowNo, df, cnt)
        return True
    else:
        return False


# In[ ]:


def funcStoreData(df):
    dfStore = df
    return dfStore


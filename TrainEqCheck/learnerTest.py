

import pandas as pd
import csv as cv
import numpy as np
import graphviz
import sys
sys.path.append('../')
from TrainEqCheck import BalCheck, DecTreeEqui, SVMEqui, AdaBoostEqui, LogRegEqui, MLPEqui 
from random import randint
import random as rd
import sys
from tqdm import tqdm
from sklearn import tree
from joblib import dump, load
import math
import os



#function to search for duplicate test data
def binSearch(alist, item):
    if(len(alist) == 0):
        return False
    else:
        midpoint = len(alist)//2
        if(alist[midpoint] == item):
            return True
        else:
            if(item < alist[midpoint]):
                return binSearch(alist[:midpoint],item)
            else:
                return binSearch(alist[midpoint+1:],item)

def addData(X, j, row, df):
    inst= np.zeros((1, X.shape[1]))
    tempMatrixMod = np.zeros((1, df.shape[1]))
    dfMod = pd.read_csv('ModifiedDataBase.csv')
    for i in range(0, X.shape[1]):
        inst[0][i] = X[j][i]   
    for i in range(0, df.shape[1]):
        for k in range(0, df.shape[1]):
            if(dfMod.columns.values[i] == df.columns.values[k]):
                tempMatrixMod[0][i] = inst[0][k]
                break
    with open('TestingDataTrainMod.csv', 'a', newline='') as csvfile:
        writer = cv.writer(csvfile)
        writer.writerows(tempMatrixMod)       
            
            
#Function to generate test data from training data
def generateTest(tst_pm, data, df, meta_op):
    dfMod = pd.read_csv('ModifiedDataBase.csv')
    testMatrix = np.zeros(((tst_pm+1), df.shape[1])) 
    
    flg = True
    testCount = 0   
    ratioTrack = []
    noOfRows = df.shape[0]

    if(meta_op == 'Column permutation'):    
        with open('TestingDataTrainMod.csv', 'w', newline='') as csvfile:
            fieldnames = dfMod.columns.values
            writer = cv.writer(csvfile)
            writer.writerow(fieldnames) 
    
    while(testCount <= tst_pm):    
        ratio = rd.randint(0, noOfRows-1)
        if(testCount >= 1):    
            flg = binSearch(ratioTrack, ratio)
            if(flg == False):
                ratioTrack.append(ratio)    
                testMatrix[testCount] = data[ratio]
                if(meta_op == 'Column permutation'):
                    addData(data, ratio, testCount, df)
                testCount = testCount +1
        if(testCount == 0):
            ratioTrack.append(ratio)     
            testMatrix[testCount] = data[ratio]
            if(meta_op == 'Column permutation'):
                addData(data, ratio, testCount, df)
            testCount = testCount +1      

    with open('TestingDataTrain.csv', 'w', newline='') as csvfile:
        fieldnames = df.columns.values
        writer = cv.writer(csvfile)
        writer.writerow(fieldnames)
        writer.writerows(testMatrix)



def genRandInst(df, meta_op):
    dfMod = pd.read_csv('ModifiedDataBase.csv')
    tempMatrix = np.zeros((1, df.shape[1]-1))
    tempMatrixMod = np.zeros((1, df.shape[1]-1))
    min_feature_val = np.zeros((df.shape[1]-1, ))
    max_feature_val = np.zeros((df.shape[1]-1, ))
    #Getting the maximum and minimum feature values for each features which is used to generate valid test data
    for i in range(0, df.shape[1]-1):
        min_feature_val[i] = df.iloc[:, i].min()
        max_feature_val[i] = df.iloc[:, i].max()

    #Generating the first test instance (x) of the pair, refer to line 3 of ranTest algo
    for i in range(0, df.shape[1]-1): 
        fe_type = df.dtypes[i]
        fe_type = str(fe_type)
        if('int' in fe_type):
            tempMatrix[0][i] = rd.randint(min_feature_val[i], max_feature_val[i])
        else:
            tempMatrix[0][i] = rd.uniform(min_feature_val[i], max_feature_val[i])
    
    for i in range(0, df.shape[1]-1):
        for j in range(0, df.shape[1]-1):
            if(dfMod.columns.values[i] == df.columns.values[j]):
                tempMatrixMod[0][i] = tempMatrix[0][j]
                break
    
    if(meta_op == 'Column permutation'):
        return tempMatrix, tempMatrixMod
    else:
        return tempMatrix



def chkInstExist(tempMatrix, df):
    firstTest = np.zeros((df.shape[1]-1, ))
    for i in range(0, df.shape[1]-1):
        firstTest[i] = tempMatrix[0][i]        
    firstTestList = firstTest.tolist()
    dfT = pd.read_csv('TestingData.csv')
    tstMatrix = dfT.values
    testMatrixList = tstMatrix.tolist()    
    for i in range(0, len(testMatrixList)-1):
        if(firstTestList == testMatrixList[i]):
            return True

    return False  


def makeMeMo4ClPm(arrMod, arrMain, df, dfMod):
    for i in range(0, df.shape[1]-1):
        for j in range(0, df.shape[1]-1):
            if(dfMod.columns.values[i] == df.columns.values[j]):
                arrMod[0][i] = arrMain[0][j]
                break
    return arrMain            


def generateMeMo(dfT, meta_op):
    n = dfT.shape[1] 
    MaxArray = np.zeros((1, n))
    MinArray = np.zeros((1, n))
    MeanArray = np.zeros((1, n))
    MedianArray = np.zeros((1, n))

    for i in range(0, n-1):
        MaxArray[0][i] = dfT.iloc[:, i].max()
        MinArray[0][i] = dfT.iloc[:, i].min()
        MeanArray[0][i] = dfT[dfT.columns[i]].mean()
        MedianArray[0][i] = dfT[dfT.columns[i]].median() 
        
    if meta_op == 'Column permutation':
        dfM = pd.read_csv('ModifiedDataBase.csv')
        MaxArray_mod = np.zeros((1, n))
        MinArray_mod = np.zeros((1, n))
        MeanArray_mod = np.zeros((1, n))
        MedianArray_mod = np.zeros((1, n))
        makeMeMo4ClPm(MaxArray_mod, MaxArray, dfT, dfM)
        makeMeMo4ClPm(MinArray_mod, MinArray, dfT, dfM)
        makeMeMo4ClPm(MeanArray_mod, MeanArray, dfT, dfM)
        makeMeMo4ClPm(MedianArray_mod, MedianArray, dfT, dfM)
        with open('TestingDataMeMoMod.csv', 'w', newline = '') as csvfile:
            fieldnames = dfM.columns.values
            writer = cv.writer(csvfile)
            writer.writerow(fieldnames)
            writer.writerows(MeanArray_mod)
            writer.writerows(MedianArray_mod)
            writer.writerows(MaxArray_mod)
            writer.writerows(MinArray_mod)

    with open('TestingDataMeMo.csv', 'a', newline = '') as csvfile:
        writer = cv.writer(csvfile)
        writer.writerows(MeanArray)
        writer.writerows(MedianArray)
        writer.writerows(MaxArray)
        writer.writerows(MinArray) 


def equivClPm(model_before, model_after, test1, test2):
    if model_before.predict(test1) != model_after.predict(test2):
        return False
    return True


def equiv(model_before, model_after, dataInstance):
    test = dataInstance
    if model_before.predict(test) != model_after.predict(test):
        return False
    return True


def reshaping(dataTest, j):
    inst = np.zeros((1, dataTest.shape[1]))
    for i in range(dataTest.shape[1]):
        inst[0][i] = dataTest[j][i]
    return inst


def equiTest(df, meta_op):
    model_before = load('model_before.joblib')
    model_after = load('model_after.joblib')
    noOfRows = 0 
    temp = 0
    count = 0
    input_count = 0
    sim_measure = 0
    flag = False
    noRow = df.shape[0]
    with open('TestingData.csv', 'w', newline='') as csvfile:
        writer = cv.writer(csvfile)
        writer.writerow(df.columns.values)
    with open('TestingDataMeMo.csv', 'w', newline='') as csvfile:
        writer = cv.writer(csvfile)
        writer.writerow(df.columns.values)  
    if meta_op == 'Column permutation':
        dfM = pd.read_csv('ModifiedDataBase.csv')
        with open('TestingDataClPm.csv', 'w', newline='') as csvfile:
            writer = cv.writer(csvfile)
            writer.writerow(dfM.columns.values)    
    
    #Function to get the input data from configuration file
    with open('ConfigDict.csv') as csv_file:
        reader = cv.reader(csv_file)
        configDict = dict(reader)
    train_ratio = float(configDict['TRAIN_RATIO'])
    MAX_INPUTS = int(configDict['MAX_INPUT'])
    no_random = MAX_INPUTS
    
    while input_count <= no_random:
        if meta_op == 'Column permutation':
            test1, test2 = genRandInst(df, meta_op)
            dataInstance = test1
        else:
            dataInstance = genRandInst(df, meta_op)

        if not chkInstExist(dataInstance, df):
            with open('TestingData.csv', 'a', newline='') as csvfile:
                writer = cv.writer(csvfile)
                writer.writerows(dataInstance)
            if meta_op == 'Column permutation':
                with open('TestingDataClPm.csv', 'a', newline='') as csvfile:
                    writer = cv.writer(csvfile)
                    writer.writerows(test2) 
                if not equivClPm(model_before, model_after, test1, test2):
                    print('The given classifier is not balanced under column permutation')
                    return True
            else:
                if not equiv(model_before, model_after, dataInstance):
                    print('The given classifier is not balanced under '+meta_op)
                    return True
            input_count = input_count + 1
    
    #Dealing with round-off error
    data = df.values
    data = np.round(data, 2)      
    training = data  
    #Test data taken from the training data
    test_param = round((noRow*train_ratio)/100) 
    generateTest(test_param, data, df, meta_op)
    generateMeMo(df, meta_op)
    dfTest_train = pd.read_csv('TestingDataTrain.csv')
    dataTest_train = dfTest_train.values
    dataTest_train = dataTest_train[:, :-1]
    
    #Checking the model with the test data generated from training data
    for i in range(0, dataTest_train.shape[0]):
        dataInstance = reshaping(dataTest_train, i)
        if meta_op == 'Column permutation':
            dfTest_trainMod = pd.read_csv('TestingDataTrainMod.csv')
            dataTest_trainMod = dfTest_trainMod.values
            dataTest_trainMod = dataTest_trainMod[:, :-1]
            test2 = reshaping(dataTest_trainMod, i)
            if(equivClPm(model_before, model_after, dataInstance, test2) == False):
                print('The given classifier is not balanced under column permutation')
                return True
        else:    
            if(equiv(model_before, model_after, dataInstance) == False):
                print('The given classifier is not balanced under '+meta_op)
                return True
           
    #Checking the model with the data generated from mean, median, max, min values of the feature
    dfMeMo = pd.read_csv('TestingDataMeMo.csv')
    dataMeMo = dfMeMo.values
    dataMeMo = dataMeMo[:, :-1]
    for i in range(0, dfMeMo.shape[0]):
        dataInstance = reshaping(dataMeMo, i)
        if meta_op == 'Column permutation':
            dfMeMoMod = pd.read_csv('TestingDataMeMoMod.csv')
            dataMeMoMod = dfMeMoMod.values
            dataMeMoMod = dataMeMoMod[:, :-1]
            test2 = reshaping(dataMeMoMod, i)
            if(equivClPm(model_before, model_after, dataInstance, test2) == False):
                print('The given classifier is not balanced under column permutation')
                return True
            
        else:
            if(equiv(model_before, model_after, dataInstance) == False):
                print('The given classifier is not balanced under '+meta_op)
                return True
    return False


def equi(model, meta_op):
    return_flag = False
    equi_computable_classifiers = {'DecTreeEqui.func_main':'DecisionTreeClassifier', 'SVMEqui.func_main':'LinearSVC',
                                   'AdaBoostEqui.func_main':'AdaBoostClassifier', 'MLPEqui.func_main':'MLPClassifier'}
    dataFile = open('Dataset.txt', 'r')
    file = dataFile.readline()
    dataFile.close()
    df = pd.read_csv(file)
    for key, classifier in equi_computable_classifiers.items():
        if classifier in str(model):
            return_flag = eval(key+'(df, meta_op)')
            return return_flag

    return equiTest(df, meta_op)        
            

def func_train(model, df):
    data = df.values
    data = np.round(data, 2)
    X = data[:, :-1]
    y = data[:, -1]
    model = model.fit(X, y)
    dump(model, 'model_before.joblib')


def func_retrain(model):
    dfM = pd.read_csv('ModifiedDataBase.csv')
    training = dfM.values
    training = np.round(training, 2)
    X = training[:, :-1]
    y = training[:, -1]
    model = model.fit(X, y)
    dump(model, 'model_after.joblib')


def func_check(model, df):
    func_train(model, df)

    with open('ConfigDict.csv') as csv_file:
        reader = cv.reader(csv_file)
        configDict = dict(reader)
    row_perm_perc = int(configDict['ROW_PERM_PER'])
    col_perm_perc = int(configDict['COL_PERM_PER'])
           
    BalCheck.funcReverseRows(df)
    func_retrain(model) 
    equi(model, 'reverse the entire dataset') 

    BalCheck.funcPermuteRowsAsc(df)
    func_retrain(model) 
    equi(model, 'Ascending class row organizing')
    
    BalCheck.funcPermuteRowsDesc(df)
    func_retrain(model) 
    equi(model, 'Descending class row organizing')
    
    BalCheck.funcPermuteRowsAlt(df)
    func_retrain(model) 
    equi(model, 'Alternating class row organizing')
    
    
    #noOfRowsPerm = round((math.factorial(df.shape[0])/100)*row_perm_perc)
    noOfRowsPerm = row_perm_perc
    for i in range(0, noOfRowsPerm):
        BalCheck.funcRandPermuteRows(df)
        func_retrain(model)  
        if equi(model, 'Random row permutation'):
            break
    
    BalCheck.funcPermRowsBat(df)
    func_retrain(model) 
    equi(model, 'Batch shuffling of rows')
    

    
    #noOfColumnsPerm = round((math.factorial(df.shape[1])/100)*col_perm_perc)
    noOfColumnsPerm = col_perm_perc
    for i in range(0, noOfColumnsPerm):
        BalCheck.funcPermuteCols(df)
        func_retrain(model)
        if equi(model, 'Column permutation'):
            break


def func_main(model):
    for i in tqdm(range(0, 9)):
        dataFile = open('Dataset.txt', 'w')
        if i == 0:
            df = pd.read_csv('DataRepository/LungCancerData.csv')
            dataFile.write('DataRepository/LungCancerData.csv')
        elif i == 1:
            df = pd.read_csv('DataRepository/BreastCancerDataWisconsinOriginal.csv')
            dataFile.write('DataRepository/BreastCancerDataWisconsinOriginal.csv')
        elif i == 2:
            df = pd.read_csv('DataRepository/CrimeData.csv')
            dataFile.write('DataRepository/CrimeData.csv')
        elif i == 3:
            df = pd.read_csv('DataRepository/GermanCreditData.csv')
            dataFile.write('DataRepository/GermanCreditData.csv')
        elif i == 4:
            df = pd.read_csv('DataRepository/ImmunotherapyData.csv')
            dataFile.write('DataRepository/ImmunotherapyData.csv')
        elif i == 5:
            df = pd.read_csv('DataRepository/AdultData.csv')
            dataFile.write('DataRepository/AdultData.csv')
        elif i == 6:
            df = pd.read_csv('DataRepository/OccupancyData.csv')
            dataFile.write('DataRepository/OccupancyData.csv')
        elif i == 7:
            df = pd.read_csv('DataRepository/SEData.csv')
            dataFile.write('DataRepository/SEData.csv') 
        elif i == 8:
            df = pd.read_csv('DataRepository/VoiceData.csv')
            dataFile.write('DataRepository/VoiceData.csv')

        dataFile.close()
        func_check(model, df)
    os.remove('model_after.joblib')
    os.remove('model_before.joblib')
    os.remove('ModifiedDataBase.csv')
    os.remove('ModifiedDataBaseTest.csv')
    os.remove('Dataset.txt')



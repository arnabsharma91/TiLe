
import pandas as pd
import numpy as np
import csv as cv
from joblib import load
import os

def func_main(df, meta_op):
    model_before = load('model_before.joblib')
    model_after = load('model_after.joblib')
    dfMod = pd.read_csv('ModifiedDataBase.csv')
    
    model_param_before = np.array(model_before.coef_)
    with open('Coef1.csv', 'w') as csvfile:
        writer = cv.writer(csvfile)
        writer.writerow(df.columns.values)
        writer.writerows(model_param_before)
    model_param_after = np.array(model_after.coef_)
    with open('Coef2.csv', 'w') as csvfile:
        writer = cv.writer(csvfile)
        writer.writerow(dfMod.columns.values)
        writer.writerows(model_param_after)
           
    dfP = pd.read_csv('Coef1.csv')
    dfN = pd.read_csv('Coef2.csv')

    for i in range (0, dfP.shape[1]):
        for j in range (0, dfN.shape[1]):
            if(dfP.columns[i] == dfN.columns[j]):
                param_diff = abs(dfP.iloc[0][i] - dfN.iloc[0][j])
                if(param_diff > 0.01):
                    print('The given classifier is not balanced under '+meta_op)
                    os.remove('Coef1.csv')
                    os.remove('Coef2.csv')
                    return True
    os.remove('Coef1.csv')
    os.remove('Coef2.csv')            
    return False            


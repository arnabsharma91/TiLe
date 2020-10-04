
import pandas as pd
import numpy as np
import csv as cv
from joblib import load

def func_main(df, meta_op):
    model_before = load('model_before.joblib')
    model_after = load('model_after.joblib')
    dfMod = pd.read_csv('ModifiedDataBase.csv')
    param_count = 0
    bias_count = 0
    
    for i in range(0, len(model_before.coefs_)):
        for j in range(0, len(model_before.coefs_[i])):
            for k in range(0, len(model_before.coefs_[i][j])):
                param_diff = abs(model_before.coefs_[i][j][k] - model_after.coefs_[i][j][k])
                if(param_diff > 0.01):
                    param_count = param_count +1
            if(param_count > len(model_before.coefs_[i][j])):                
                print('The given Classifier is not balanced under '+meta_op)
                return True
    for i in range(0, len(model_before.intercepts_)):
        for j in range(0, len(model_before.intercepts_[i])):
            bias_diff = abs(model_before.intercepts_[i][j] - model_after.intercepts_[i][j])
            if(bias_diff > 0.01):
                bias_count = bias_count+1
        if(bias_count > len(model_before.intercepts_[i])):    
            print('The given Classifier is not balanced under '+meta_op)
            return True
    return False    
        


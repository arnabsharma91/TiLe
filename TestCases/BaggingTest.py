
# coding: utf-8

# In[1]:


import pandas as pd
import csv as cv
#K-NN classifier
#Import Library
from sklearn.ensemble import BaggingClassifier

import numpy as np

import fileinput

import graphviz
import TrainData


# In[ ]:


model = BaggingClassifier(base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, 
                          bootstrap=True, bootstrap_features=False, oob_score=False, 
                          warm_start=False, n_jobs=1, random_state=1, verbose=0)

TrainData.funcTrainTest(model)


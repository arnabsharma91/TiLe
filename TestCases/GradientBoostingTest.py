
# coding: utf-8

# In[2]:


import pandas as pd
import csv as cv
#K-NN classifier
#Import Library
from sklearn.ensemble import GradientBoostingClassifier

import numpy as np

import fileinput

import graphviz
import TrainData


# In[ ]:


model = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=500, subsample=1.0, 
                                   criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, 
                                   min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, 
                                   min_impurity_split=None, init=None, random_state=1, max_features=None, verbose=0, 
                                   max_leaf_nodes=None, warm_start=False, presort='auto')

TrainData.funcTrainTest(model)



# coding: utf-8

# In[1]:


import pandas as pd
import csv as cv
#K-NN classifier
#Import Library
from sklearn.linear_model import SGDClassifier

import numpy as np

import fileinput

import graphviz
import TrainData


# In[ ]:


model = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=None, 
                      tol=None, shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=1, 
                      learning_rate='optimal', eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, 
                      average=False, n_iter=None)

TrainData.funcTrainTest(model)


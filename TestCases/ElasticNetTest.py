
# coding: utf-8

# In[1]:


import pandas as pd
import csv as cv
#K-NN classifier
#Import Library
from sklearn.linear_model import ElasticNet

import numpy as np

import fileinput

import graphviz
import TrainData


# In[ ]:


model = ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, 
                   copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=1, selection='cyclic')


TrainData.funcTrainTest(model)


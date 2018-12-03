
# coding: utf-8

# In[2]:


import pandas as pd
import csv as cv
#K-NN classifier
#Import Library
from sklearn.linear_model import LinearRegression

import numpy as np

import fileinput

import graphviz
import TrainData


# In[ ]:


model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)

TrainData.funcTrainTest(model)


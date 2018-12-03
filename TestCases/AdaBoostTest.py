
# coding: utf-8

# In[1]:


import pandas as pd
import csv as cv
#K-NN classifier
#Import Library
from sklearn.ensemble import AdaBoostClassifier

import numpy as np

import fileinput

import graphviz
import TrainData


# In[ ]:


model = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', 
                           random_state=1)

TrainData.funcTrainTest(model)


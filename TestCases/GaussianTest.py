
# coding: utf-8

# In[5]:


import pandas as pd
import csv as cv
#K-NN classifier
#Import Library
from sklearn.gaussian_process import GaussianProcessClassifier

import numpy as np

import fileinput

import graphviz
import TrainData


# 

# In[ ]:


model = GaussianProcessClassifier(kernel=None, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, 
                                  max_iter_predict=100, warm_start=False, copy_X_train=True, random_state=1, 
                                  multi_class='one_vs_rest', n_jobs=1)

TrainData.funcTrainTest(model)


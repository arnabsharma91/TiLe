
# coding: utf-8

# In[1]:


import pandas as pd
import csv as cv
#K-NN classifier
#Import Library
from sklearn.ensemble import ExtraTreesClassifier

import numpy as np

import fileinput

import graphviz
import TrainData


# In[ ]:


model = ExtraTreesClassifier(n_estimators=10, criterion='entropy', max_depth=None, min_samples_split=2, 
                             min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', 
                             max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, 
                             bootstrap=False, oob_score=False, n_jobs=1, random_state=1, verbose=0, 
                             warm_start=False, class_weight=None)

TrainData.funcTrainTest(model)


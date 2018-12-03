
# coding: utf-8

# In[2]:


import pandas as pd
import csv as cv
#K-NN classifier
#Import Library
from sklearn.ensemble import VotingClassifier

import numpy as np

import fileinput

import graphviz
import TrainData


# In[ ]:


model = VotingClassifier(voting='hard', weights=None, n_jobs=1, flatten_transform=None)

TrainData.funcTrainTest(model)



# coding: utf-8

# In[1]:


import pandas as pd
import csv as cv
#K-NN classifier
#Import Library
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import numpy as np
from sklearn.naive_bayes import BaseNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import fileinput
import TrainData


# In[ ]:


model = RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                                   min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, 
                                   min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, 
                                   n_jobs=1, random_state=1, verbose=0)
TrainData.funcTrainTest(model)                               
 



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
import fileinput
import TrainData


# In[ ]:


model = DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=20, min_samples_split=2, 
                         min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=3, random_state=1, 
                         max_leaf_nodes=20, min_impurity_decrease=0.0, min_impurity_split=None, class_weight="balanced", 
                               presort=False)

TrainData.funcTrainTest(model)
   


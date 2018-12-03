
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
from sklearn.linear_model import LogisticRegression
import fileinput
from sklearn.neural_network import MLPClassifier
import TrainData


# In[ ]:


model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, 
                                    intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear')
TrainData.funcTrainTest(model)
    


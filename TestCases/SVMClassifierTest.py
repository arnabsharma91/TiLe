
# coding: utf-8

# In[2]:


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
import graphviz
import matplotlib
import matplotlib.pyplot as plt
import TrainData


# In[ ]:


model = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', 
        fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=1, max_iter=1000)

TrainData.funcTrainTest(model)



# coding: utf-8

# In[51]:


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
import TrainData


# In[1]:


model = GaussianNB()
TrainData.funcTrainTest(model)               


# In[ ]:






# coding: utf-8

# In[6]:


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


# In[7]:


#training the model
model = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                           beta_1=0.9, beta_2=0.999, early_stopping=False,
                           epsilon=1e-08, hidden_layer_sizes=(5,5), learning_rate='constant',
                           learning_rate_init=0.001, max_iter=200, momentum=0.9,
                           nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
                           solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
                           warm_start=False)

TrainData.funcTrainTest(model)
                
                
 


# In[ ]:



 


# In[ ]:





# In[ ]:





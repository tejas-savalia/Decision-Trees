
# coding: utf-8

# In[284]:

import numpy
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[285]:

def load_dataset(url):
    dataset = numpy.genfromtxt(url, dtype = 'str', delimiter = ',')
    X, y = dataset[:, :-1], dataset[:, -1]
    return X, y


# In[286]:

def preprocessing(X, y):
    X_new = numpy.zeros(numpy.shape(X.T))
    y_new = numpy.zeros(numpy.shape(y.T))
    row = 0
    for i in range(len(X.T)):
        X_new[i] = numerize(X.T[i])
    y_new = numerize(y)    
    return X_new.astype(int).T, y_new.astype(int).T


# In[287]:

def numerize(feature):
    unique = numpy.unique(feature)
    numerized = numpy.zeros(numpy.shape(feature))
    for i in range(len(unique)):
        numerized[feature == unique[i]] = i
    return numerized


# In[288]:

X, y = load_dataset("E:\Lecs\IIIT\SMAI\Assignments\Assignment 6\car.txt")


# In[289]:

def Entropy(y):
    countClasses = numpy.bincount(y)
    countClasses += 1
    class_ratio = countClasses/float(len(y))
    entropy = sum(-numpy.log2(class_ratio)*class_ratio)
    return entropy


# In[290]:

X, y = preprocessing(X, y)


# In[291]:

def Gain(X, y, feature):
    negative = 0
    attribute_values = numpy.unique(feature)
    for i in attribute_values:
        y_new = y[feature == i]
        negative += Entropy(y_new) * (len(y_new)/float(len(y)))
    gain = Entropy(y_new) - negative
    return gain


# In[292]:

def select_attribute(X, y):
    gain = list()
    for i in range(len(X.T)):
        gain.append(Gain(X, y, X.T[i]))
    gain = numpy.array(gain)
    return numpy.argmax(gain)


# In[293]:

def new_space(X, y):
    attr = select_attribute(X, y)
    print "Enter value of attribute ", attr, ": "
    value = int(input())
    X_new = X[X.T[attr] == value]
    y_new = y[X.T[attr] == value]
    X_new = numpy.delete(X_new, attr, axis = 1)
    return X_new, y_new


# In[294]:

def decision_trees(X, y):
    classes = numpy.shape(X)[1]
    print classes
    for i in range(classes):
        X, y = new_space(X, y)
        print numpy.unique(y)
        if len(numpy.unique(y)) == 1:
            return numpy.unique(y)
    return numpy.unique(y)


# In[295]:

print decision_trees(X, y)


# In[ ]:




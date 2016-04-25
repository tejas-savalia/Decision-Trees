
# coding: utf-8

# In[5]:

from sklearn import svm
from sklearn import datasets
from sklearn.metrics import *
from sklearn import cross_validation
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold
import numpy
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[6]:

X = numpy.genfromtxt("E:\Lecs\IIIT\SMAI\Assignments\Assignment 6\\arcene_train.data.txt", dtype = 'float', delimiter = ' ')
y = numpy.genfromtxt("E:\Lecs\IIIT\SMAI\Assignments\Assignment 6\\arcene_train.labels.txt", dtype = 'float', delimiter = ' ')
pca = PCA(n_components = 10)
pca.fit(X)
arcene_pca = pca.transform(X)


# In[7]:

clf = svm.SVC(kernel = 'linear')
clf.fit(arcene_pca[:20, :], y[:20])
scores = cross_validation.cross_val_score(clf, arcene_pca[:10, :], y[:10], cv = 5)
print scores


# In[8]:

predicted = clf.predict(arcene_pca[(-len(y)/5):, :])
print "Precision: ", precision_score(y[-(len(y)/5):], predicted)
print "Recall: ", recall_score(y[-(len(y)/5):], predicted)


# In[11]:

clf = svm.SVC(kernel = 'rbf')
clf.fit(arcene_pca, y)
scores = cross_validation.cross_val_score(clf, arcene_pca[:80, :], y[:80], cv = 5)
print scores


# In[13]:

predicted = clf.predict(arcene_pca[80:, :])
print "Precision: ", precision_score(y[80:], predicted)
print "Recall: ", recall_score(y[80:], predicted)


# In[17]:

pca = PCA(n_components = 100)
pca.fit(X)
arcene_pca = pca.transform(X)
clf = svm.SVC(kernel = 'linear')
clf.fit(arcene_pca, y)
scores = cross_validation.cross_val_score(clf, arcene_pca, y, cv = 5)
print scores


# In[18]:

predicted = clf.predict(arcene_pca[80:, :])
print "Precision: ", precision_score(y[80:], predicted)
print "Recall: ", recall_score(y[80:], predicted)


# In[19]:

pca = PCA(n_components = 100)
pca.fit(X)
arcene_pca = pca.transform(X)
clf = svm.SVC(kernel = 'rbf')
clf.fit(arcene_pca, y)
scores = cross_validation.cross_val_score(clf, arcene_pca, y, cv = 5)
print scores


# In[20]:

predicted = clf.predict(arcene_pca[80:, :])
print "Precision: ", precision_score(y[80:], predicted)
print "Recall: ", recall_score(y[80:], predicted)


# In[21]:

digits = datasets.load_digits()
pca = PCA(n_components = 3)
pca.fit(digits.data)
digits_pca = pca.transform(digits.data)
clf = svm.SVC(kernel = 'linear')
clf.fit(digits_pca, digits.target)
scores = cross_validation.cross_val_score(clf, digits_pca, digits.target, cv = 5)
print scores


# In[23]:

predicted = clf.predict(digits_pca[80:, :])
print "Precision: ", precision_score(digits.target[80:], predicted)
print "Recall: ", recall_score(digits.target[80:], predicted)


# In[24]:

clf = svm.SVC(kernel = 'rbf')
clf.fit(digits_pca, digits.target)
scores = cross_validation.cross_val_score(clf, digits_pca, digits.target, cv = 5)
print scores


# In[25]:

clf = svm.SVC(kernel = 'rbf')
clf.fit(digits_pca[:-(len(digits.target)/5), :], digits.target[:-(len(digits.target)/5)])
predicted = clf.predict(digits_pca[(len(digits.target)/5):, :])
print "Precision: ", precision_score(digits.target[(len(digits.target)/5):], predicted)
print "Recall: ", recall_score(digits.target[(len(digits.target)/5):], predicted)


# In[26]:

pca = PCA(n_components = 7)
pca.fit(digits.data)
digits_pca = pca.transform(digits.data)
clf = svm.SVC(kernel = 'linear')
clf.fit(digits_pca, digits.target)
scores = cross_validation.cross_val_score(clf, digits_pca, digits.target, cv = 5)
print scores


# In[28]:

predicted = clf.predict(digits_pca[80:, :])
print "Precision: ", precision_score(digits.target[80:], predicted)
print "Recall: ", recall_score(digits.target[80:], predicted)


# In[29]:

clf = svm.SVC(kernel = 'rbf')
clf.fit(digits_pca, digits.target)
scores = cross_validation.cross_val_score(clf, digits_pca, digits.target, cv = 5)
print scores


# In[30]:

clf = svm.SVC(kernel = 'rbf')
clf.fit(digits_pca[:-(len(digits.target)/5), :], digits.target[:-(len(digits.target)/5)])
predicted = clf.predict(digits_pca[(len(digits.target)/5):, :])
print "Precision: ", precision_score(digits.target[(len(digits.target)/5):], predicted)
print "Recall: ", recall_score(digits.target[(len(digits.target)/5):], predicted)


# In[44]:

print accuracy


# In[ ]:




# In[ ]:




# In[ ]:




# In[220]:

def load_wineData(url):
    loaded = numpy.genfromtxt(url, delimiter = ',', dtype = "float")
    numpy.random.shuffle(loaded)
    X, y = loaded[:, 1:], loaded[:, 0]
    return X.astype(float), y.astype(float)


# In[221]:

X, y = load_wineData("E:\Lecs\IIIT\SMAI\Assignments\Assignment 6\\wine.data.txt")


# In[222]:

print numpy.shape(X)


# In[223]:

PCA_transformed_XTrain = PCA(X, 3)
print numpy.shape(PCA_transformed_XTrain[len(PCA_transformed_XTrain)/5:])
print numpy.shape(y[len(PCA_transformed_XTrain)/5:])


# In[224]:

clf = svm.SVC(kernel = 'linear')
clf.fit(PCA_transformed_XTrain[len(PCA_transformed_XTrain)/5:], y[len(PCA_transformed_XTrain)/5:])
predicted = clf.predict(PCA_transformed_X[:len(X)/5, :])

print "Accuracy: ", accuracy_score(y[:len(PCA_transformed_XTrain)/5], predicted)
print "Precision: ", precision_score(y[:len(PCA_transformed_XTrain)/5], predicted)
print "Recall: ", recall_score(y[:len(PCA_transformed_XTrain)/5], predicted)

clf = svm.SVC(kernel = 'rbf', C = 1000)
clf.fit(PCA_transformed_XTrain[len(PCA_transformed_XTrain)/5:], y[len(PCA_transformed_XTrain)/5:])
predicted = clf.predict(PCA_transformed_X[:len(X)/5, :])

print "Accuracy: ", accuracy_score(y[:len(PCA_transformed_XTrain)/5], predicted)
print "Precision: ", precision_score(y[:len(PCA_transformed_XTrain)/5], predicted)
print "Recall: ", recall_score(y[:len(PCA_transformed_XTrain)/5], predicted)


# In[ ]:




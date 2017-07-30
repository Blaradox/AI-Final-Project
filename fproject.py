#!/usr/bin/env python3
"""
Make sure to fill in the following information before submitting your
assignment. Your grade may be affected if you leave it blank!
File name: fproject.py
Author username(s): sanuk, sloaneat 
Date: Dec 16, 2015
Submission name: Final Project
"""

import numpy as np
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import sys

# http://stackoverflow.com/questions/19486369/extract-csv-file-specific-columns-to-list-in-python

# colnames = ['year', 'name', 'city', 'latitude', 'longitude']
# data = pandas.read_csv('test.csv', names=colnames)
# If you want your lists as in the question, you can now do:

# names = data.name.tolist()
# latitude = data.latitude.tolist()
# longitude = data.longitude.tolist()

# # An array X of size [n_samples, n_features] holding the training samples
# X = [[0, 0], [1, 1]]
# # An array y of class labels (strings or integers), size [n_samples]
# y = [0, 1] 

# clf = svm.SVC()
# clf.fit(X, y)  
# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
#     max_iter=-1, probability=False, random_state=None, shrinking=True,
#     tol=0.001, verbose=False)

print(sys.argv[0])
if len(sys.argv) < 2:
    ifname = input("Please input a training file: ")
else:
    ifname = sys.argv[1]

print("....Training with " + ifname)
fp = open(ifname, 'r')
       
# Read first line to make list of attribute
colnames = fp.readline().strip().split(',')

# Read rest of csv for training samples to make an array of size [n_samples, n_features] 
samples = []
for line in fp:
    sample = []
    for val in line.strip().split(','):
        sample.append(val)
    samples.append(sample)

# split 'samples' into 'X' and 'y'
outcomes = []
for line in samples:
    outcome = line.pop()
    outcomes.append(outcome)
X = np.array(samples)
y = np.array(outcomes)
X.reshape((306,3))
y.reshape((306,1))

# split into a training and testing set
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# The default parameters for SVC are a radial basis function kernel of degree 3      
model = svm.SVC(C=1.0, cache_size=1000, class_weight=None, coef0=0.0,
     decision_function_shape=None, degree=2, gamma='auto', kernel='rbf',
     max_iter=-1, probability=False, random_state=None, shrinking=True,
     tol=0.00001, verbose=False) 
model.fit(X_train, y_train.ravel())  

expected = y_test
predicted = model.predict(X_test)

print(classification_report(expected, predicted, target_names=[">=5 years","<5 years"]))

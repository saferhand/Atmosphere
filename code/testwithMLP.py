# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:14:47 2019

@author: Saferhand
"""

import strlearn
from sklearn import neural_network

clf = neural_network.MLPClassifier()
X, y = strlearn.utils.load_arff('//results/toyset.arff')
learner = strlearn.Learner(X, y, clf)
learner.run()
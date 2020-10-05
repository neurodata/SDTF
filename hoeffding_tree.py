'''
Main Author: Haoyin Xu
Corresponding Email: haoyinxu@gmail.com
'''
# import the necessary packages
import numpy as np
import pandas as pd
import matplotlib as plt

from sklearn import tree

# define the Hoeffding Decision Tree
def hdt(X, y):
    model = tree.DecisionTreeClassifier()
    model.fit(X, y)

    tree.plot_tree(model,filled=True)

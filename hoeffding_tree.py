"""
Main Author: Haoyin Xu
Corresponding Email: haoyinxu@gmail.com
"""
# import the necessary packages
from sklearn import tree

# define the Hoeffding Decision Tree
def hdt(X, y):
    model = tree.DecisionTreeClassifier()
    model.fit(X, y)

    tree.plot_tree(model, filled=True)

# -*- coding: utf-8 -*-


""" Main Application part"""

from EntropyClassifier.Classifier import P2
from sklearn import metrics
from sklearn.datasets import make_circles, make_moons
from matplotlib.colors import  ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
import math
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import hamming_loss, accuracy_score 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

  
def mytrain_test_split(X,y, train_size): 
    X_train = X.sample(frac = train_size, random_state=2021)
    X_test = X.drop(X_train.index)
    y_train = y.sample(frac = train_size,random_state=2021)
    y_test = y.drop(y_train.index) 
    label = str(y_train.columns.values[0])
    y_test = y_test[label].values.tolist()
    return X_train, X_test, y_train, y_test


def plot_clf(model, y_test, X_test, activationname):
    if activationname == 'heaviside':
        treshhold = 0
    h = .01  # step size in the mesh
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
    y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    x_new = np.c_[xx.ravel(), yy.ravel()]
    print("te")
    #x_new = pd.DataFrame(x_new, columns=['x1','x2'])
    Z = model.predict(x_new,treshhold)
    print("d")
    Z =  np.array( Z)
    Z = Z.reshape(xx.shape)
    # Put the result into a color plot
    fig = plt.figure()
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("2-Class classification")
    return fig

"""--------------------------------------Data Preparation-------------------------------------------------"""
train_val = 70/100
X, y = make_moons(n_samples=200, shuffle=True, noise=0.2, random_state=1)

X = pd.DataFrame(X)
y = pd.DataFrame(y, columns=['classe'])
X_train, X_test, y_train, y_test = mytrain_test_split(X,y, train_val)


X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.fit_transform(X_test)
"""--------------------------------------------------------------------------------------------------------"""


print("Beginning P2 Classification")
model = P2(activation='heaviside', optimizer="montecarlo") 
best_params_calc,list_S, list_r = model.run(X_train, y_train, X_train, y_train)

model1.tree.printTreer()
plot_clf(model, y_train, X_train, activationname='heaviside')

#arr = np.array([0.954831, -0.134719])
#caminho = model.path(arr, 0)
#leaf = model.leaf(arr,0)
#final = model.leaf(arr, 0)
#Arvore = model.tree
"""-----------------------------------------------------Plot and Tree rendering Stuff-----------------------------"""
#plot_clf(model, y_train, X_train, activationname='heaviside')



# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 12:41:43 2016

@author: Kiyoko
"""

###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
###########################################
import numpy as np
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve

from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import roc_auc_score, make_scorer


# Map locations on the map of Chicago
def Map(df):
    mapdata = np.loadtxt("../input/mapdata_copyright_openstreetmap_contributors.txt")
    
    aspect = mapdata.shape[0] * 1.0 / mapdata.shape[1]
    lon_lat_box = (-88, -87.5, 41.6, 42.1)
    
    plt.figure(figsize=(10,14))
    plt.imshow(mapdata, 
               cmap=plt.get_cmap('gray'), 
               extent=lon_lat_box, 
               aspect=aspect)
    
    plt.scatter(df['Longitude'],df['Latitude'], s=df['NumMosquitos']*2, edgecolors='b', facecolors='None')
    stations = np.array([[-87.933, 41.995], [-87.752, 41.786]])
    for i in range(2):
        plt.plot(stations[i,0], stations[i,1], '^', color='r')
    plt.xlim([-88, -87.5])
    plt.ylim([41.6, 42.1])
    plt.savefig('map.png')

# plot learning curves
def LearningCurve(clf, features, labels):
    cv = ShuffleSplit(features.shape[0], n_iter = 10, random_state = 0)
    train_sizes = np.linspace(0.1, 1.0, 10)
    scorer = make_scorer(roc_auc_score)
    
    sizes, train_scores, test_scores = learning_curve(clf, features, labels, train_sizes = train_sizes, cv=cv, scoring=scorer)

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.plot(sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.fill_between(sizes, train_scores_mean - train_scores_std, \
                         train_scores_mean + train_scores_std, alpha=0.1, \
                         color="r")
    plt.fill_between(sizes, test_scores_mean - test_scores_std, \
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.legend(loc='best')
        
# plot percent positive per group
def PercPos(fig, df, groups):
    for i in range(len(groups)):
        group = df.groupby([groups[i]])
        percent_positive_per_group = dict(group['WnvPresent'].sum() / group['WnvPresent'].count() * 100.0)
        
        charts = fig.add_subplot(2, 3, i+1)
        charts.set_xlabel(groups[i])
        charts.set_ylabel('% Virus Present')
        charts.set_xticks(range(len(percent_positive_per_group)))
        charts.set_xticklabels(percent_positive_per_group.keys(), rotation=70)
        charts.bar(range(len(percent_positive_per_group)), percent_positive_per_group.values(), align = 'center')
        locs, labels = plt.xticks()
        

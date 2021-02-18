import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn import tree
from sklearn.feature_selection import SelectPercentile, f_classif,VarianceThreshold
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel


########### --- DATASET PROCESS --- ###########
ROOT = os.path.abspath(os.path.dirname('sentient_things'))
print(ROOT)
DATA_LOCATION = os.path.join(ROOT, 'sentient_things/data/raw/encuesta_GS_v3_blue.csv')
print(DATA_LOCATION)

with open(DATA_LOCATION, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)

dataset = np.genfromtxt(DATA_LOCATION, delimiter=',', 
        skip_header=1)

features_tmp = []
features = dataset[0:,[range(18)]]
#features = dataset[0:,[range(14)]]

ratings_tmp = []
ratings = dataset[0:, [range(18,38)]]
#ratings = dataset[0:, [range(14,35)]]


for i in range(len(features)):
    if not np.isnan(dataset[i]).any():
        features_tmp.append(list(features[i][0]))
        ratings_tmp.append(list(ratings[i][0]))

features = np.asarray(features_tmp)
print(features.shape)
ratings = np.asarray(ratings_tmp)

########### --- END DATASET PROCESS --- ###########

nb_epochs = 1.0

# Ninteen zeros to ensure the data fits.
et_total = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
f_total = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
clf_total = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

for i in range(int(nb_epochs)):
    extraTrees = ExtraTreesClassifier(n_estimators=1, n_jobs=-1)
    extraTrees.fit(features,ratings)
    et_importances = extraTrees.feature_importances_
    et_total = [x + y for x, y in zip(et_importances, et_total)]
    
    forest = RandomForestClassifier(n_estimators=1, n_jobs=-1)
    forest.fit(features,ratings)
    f_importances = forest.feature_importances_
    f_total = [x + y for x, y in zip(f_importances, f_total)]
    
    clf = DecisionTreeClassifier(min_samples_split=0.1,
                                  max_features='auto',
                                  )
    clf.fit(features,ratings)
    clf_importances = clf.feature_importances_
    clf_total = [x + y for x, y in zip(clf_importances, clf_total)]
    print("%i epoch out of %i" % (i+1,int(nb_epochs)))
        
et_total = [x / nb_epochs for x in et_total]
f_total = [x / nb_epochs for x in f_total]
clf_total = [x / nb_epochs for x in clf_total]

et_feature_pairs = dict(zip(headers, et_total))
f_feature_pairs = dict(zip(headers, f_total))
clf_feature_pairs = dict(zip(headers, clf_total))


print("\n---- EXTRA TREES CLASSIFIER ----")
i=1
for key in sorted(et_feature_pairs, key=et_feature_pairs.get, reverse=True):
        print('%d. feature %s (%f)' % (i, key, et_feature_pairs[key]))
        i += 1
print("\n---- RANDOM FOREST CLASSIFIER ----")
i=1
for key in sorted(f_feature_pairs, key=f_feature_pairs.get, reverse=True):
        print('%d. feature %s (%f)' % (i, key, f_feature_pairs[key]))
        i += 1
print("\n---- DECISION TREE CLASSIFIER ----")
i=1
for key in sorted(clf_feature_pairs, key=clf_feature_pairs.get, reverse=True):
        print('%d. feature %s (%f)' % (i, key, clf_feature_pairs[key]))
        i += 1


import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
import csv

train_x = pd.read_csv('Dataset/train/features.csv', header = None)
train_y = pd.read_csv('Dataset/train/Labels.csv', header = None)
test_x = pd.read_csv('Dataset/test/features.csv', header = None)
test_y = pd.read_csv('Dataset/test/Labels.csv', header = None)

classifier = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1), n_estimators = 1000, learning_rate = 0.1)
classifier.fit(train_x, train_y)
selected = classifier.feature_importances_


wtr = csv.writer(open ('out.csv', 'w'), delimiter=',', lineterminator='\n')
for x in selected : wtr.writerow ([x])
print(selected.shape)

counter = 0
for i in selected:
        print(i)
        counter += 1

print(counter)
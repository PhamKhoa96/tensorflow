# -*- coding: utf-8 -*-
"""
Created on Sun May 24 10:45:03 2020

@author: phamk
"""


# Ignore warnings 

import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784')

X_data, y_data = mnist.data, mnist.target

%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib

plt.imshow(X_data[2].reshape(28, 28))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

import seaborn as sns

nb_imgs = 10

fig, axes = plt.subplots(nb_imgs, nb_imgs, figsize=(15, 15))

for i in range(nb_imgs):
    for j in range(nb_imgs):
        axes[i][j].imshow(X_test[i*j].reshape(28, 28))
        
y_test_5 = (y_test == 5)
y_train_5 = (y_train == 5)

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)

sgd_clf.fit(X_train, y_train_5)

plt.imshow(X_test[15].reshape(28, 28))

sgd_clf.predict([X_test[15]])

#Performance measure
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=5, scoring='accuracy')

# Classification report
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)

# Precision Not number 5
precision_not_number_5 = 48763 / (705 + 48763)

# Precision number 5
precision_number_5 = 4322 / (4322 + 2210)

# Recall Not number 5
recall_not_number_5 = 48763 / (48763 + 2210)

# Recall Number 5
recall_number_5 = 4322 / (4322 + 705)

# F1 Score 2*precision*recall / (precision + recall)
def f1_score(precision, recall):
    return 2*precision*recall / (precision + recall)

f1_score(precision_number_5, recall_number_5)

from sklearn.metrics import classification_report
target_names = ['Not number 5', 'Number 5']

print(classification_report(y_train_5, y_train_pred, target_names=target_names))
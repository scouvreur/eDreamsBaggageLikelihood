"""
================================================

"All models are wrong, but some are useful."
George Box, 1976

================================================

A binary classifier for baggage likelihood
prediction using gradient boosted machines

================================================
"""

print(__doc__)

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score

test = pd.read_csv("test_xgboost.csv", sep = ",")
train = pd.read_csv("train_xgboost.csv", sep = ",")

categorical_features = ["IS_ALONE","HAUL_TYPE","DEVICE","TRIP_TYPE","COMPANY","DISTANCE_CAT"]
numerical_features = ["FAMILY_SIZE"]

# Conversion of string categorical variables to encoded labels
for feature in categorical_features:
	train[feature] = LabelEncoder().fit_transform(train[feature].astype("str"))
	test[feature] = LabelEncoder().fit_transform(test[feature].astype("str"))

train["EXTRA_BAGGAGE"] = LabelEncoder().fit_transform(train["EXTRA_BAGGAGE"])

features = categorical_features + numerical_features

# Separation of features and outcome
X_train = train[list(features)].values
Y_train = train["EXTRA_BAGGAGE"].values
X_test = test[list(features)].values

# Train/validation split to compute F1 and AUC internally without holdout set
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train,
																test_size=0.2,
																random_state=747)

# Rule-ot-thumb XGBoost parameters
n_estimators=300
max_depth=3
learning_rate=0.1

clf = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
clf.fit(X_train, Y_train)

Y_test = clf.predict(X_test)
Y_test_proba = clf.predict_proba(X_test)

print("--- Model parameters ---")
print("XGBClassifier(n_estimators={}, max_depth={}, learning_rate={})".format(n_estimators, max_depth, learning_rate))

print("--- Validation set ---")
print("AUC;{}".format(roc_auc_score(Y_validation, clf.predict(X_validation))))
print("F1;{}".format(f1_score(Y_validation, clf.predict(X_validation), average='micro')))

print("--- Training set ---")
print("AUC;{}".format(roc_auc_score(Y_train, clf.predict(X_train))))
print("F1;{}".format(f1_score(Y_train, clf.predict(X_train), average='micro')))

f = open("submission_xgboost.csv", 'w')
f.write("ID,EXTRA_BAGGAGE\n")
for i in range(len(Y_test)):
    # f.write("{},{}\n".format(i,Y_test[i]))
    f.write("{},{:6f}\n".format(i,Y_test_proba[i,1]))

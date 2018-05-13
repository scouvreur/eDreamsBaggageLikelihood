"""
================================================

"All models are wrong, but some are useful."
George Box, 1976

================================================
"""

print(__doc__)

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

test = pd.read_csv("test_xgboost.csv", sep = ",")
train = pd.read_csv("train_xgboost.csv", sep = ",")

categorical_features = ["HAUL_TYPE","DEVICE","TRIP_TYPE","COMPANY"]

train["IS_ALONE"] = LabelEncoder().fit_transform(train["IS_ALONE"].astype("str"))
train["COMPANY"] = LabelEncoder().fit_transform(train["COMPANY"].astype("str"))
train["HAUL_TYPE"] = LabelEncoder().fit_transform(train["HAUL_TYPE"].astype("str"))
train["TRIP_TYPE"] = LabelEncoder().fit_transform(train["TRIP_TYPE"].astype("str"))
train["DEVICE"] = LabelEncoder().fit_transform(train["DEVICE"].astype("str"))
train["DISTANCE"] = train["DISTANCE"].astype("float")
train["EXTRA_BAGGAGE"] = LabelEncoder().fit_transform(train["EXTRA_BAGGAGE"].astype("str"))

test["IS_ALONE"] = LabelEncoder().fit_transform(test["IS_ALONE"].astype("str"))
test["COMPANY"] = LabelEncoder().fit_transform(test["COMPANY"].astype("str"))
test["HAUL_TYPE"] = LabelEncoder().fit_transform(test["HAUL_TYPE"].astype("str"))
test["TRIP_TYPE"] = LabelEncoder().fit_transform(test["TRIP_TYPE"].astype("str"))
test["DEVICE"] = LabelEncoder().fit_transform(test["DEVICE"].astype("str"))
test["DISTANCE"] = test["DISTANCE"].astype("float")

features = ["IS_ALONE","FAMILY_SIZE","HAUL_TYPE","DEVICE","TRIP_TYPE","COMPANY"]

X_train = train[list(features)].values
Y_train = train["EXTRA_BAGGAGE"].values
X_test = test[list(features)].values

X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train,
																test_size=0.2,
																random_state=747)

n_estimators=800
max_depth=3
learning_rate=1.0

# clf = XGBClassifier()
clf = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, reg_alpha=1, reg_lambda=1)
clf.fit(X_train, Y_train)

Y_test_binary = clf.predict(X_test)
Y_test_proba = clf.predict_proba(X_test)

print("--- Model parameters ---")
print("XGBClassifier(n_estimators={}, max_depth={}, learning_rate={})".format(n_estimators, max_depth, learning_rate))

print("--- Validation set ---")
# print("AUC;{}".format(roc_auc_score(Y_validation, clf.predict(X_validation))))
print("F1;{}".format(f1_score(Y_validation, clf.predict(X_validation))))

print("--- Training set ---")
# print("AUC;{}".format(roc_auc_score(Y_train, clf.predict(X_train))))
print("F1;{}".format(f1_score(Y_train, clf.predict(X_train))))

print("ID;EXTRA_BAGGAGE")
for i in range(len(X_test)):
    # print("{};{};{:.6f}".format(i,Y_test_binary[i],Y_test_proba[i,0]))
    print("{};{:.6f}".format(i,Y_test_proba[i,0]))

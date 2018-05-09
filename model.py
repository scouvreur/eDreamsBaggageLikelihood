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
from sklearn.linear_model import LogisticRegression

test = pd.read_csv("test.csv", sep = ";")
train = pd.read_csv("train.csv", sep = ";")

train = pd.get_dummies(data=train, columns=["HAUL_TYPE", "TRIP_TYPE"])
train["DISTANCE"] = train["DISTANCE"].astype("float")
train["EXTRA_BAGGAGE"] = LabelEncoder().fit_transform(train["EXTRA_BAGGAGE"].astype("str"))

test = pd.get_dummies(data=test, columns=["HAUL_TYPE", "TRIP_TYPE"])
test["DISTANCE"] = test["DISTANCE"].astype("float")

features = ["HAUL_TYPE_CONTINENTAL","HAUL_TYPE_DOMESTIC","HAUL_TYPE_INTERCONTINENTAL",
			"TRIP_TYPE_MULTI_DESTINATION","TRIP_TYPE_ONE_WAY","TRIP_TYPE_ROUND_TRIP"]

X_train = train[list(features)].values
Y_train = train["EXTRA_BAGGAGE"].values
X_test = test[list(features)].values

X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.2, random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)

# Y_test = logreg.predict(X_test)

# print("--- Training set ---")
# print("Accuracy;{}".format(accuracy_score(Y_train, logreg.predict(X_train))))
# print("F1;{}".format(f1_score(Y_train, logreg.predict(X_train))))
# print("AUC;{}".format(roc_auc_score(Y_train, logreg.predict(X_train))))

# print("--- Validation set ---")
# print("Accuracy;{}".format(accuracy_score(Y_validation, logreg.predict(X_validation))))
# print("F1;{}".format(f1_score(Y_validation, logreg.predict(X_validation))))
# print("AUC;{}".format(roc_auc_score(Y_validation, logreg.predict(X_validation))))

# print("Ids;TARGET")
# for i in range(len(X_test)):
# 	print("ID{};{}".format(i+26500,Y_test[i]))

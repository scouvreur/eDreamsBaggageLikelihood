"""
=============================================================

"All models are wrong, but some are useful."
George Box, 1976

=============================================================

A binary classifier for baggage likelihood prediction using
gradient boosted machines

=============================================================
"""

print(__doc__)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import roc_auc_score

from scipy import stats, integrate
import seaborn as sns
sns.set(color_codes=True)

test = pd.read_csv("test_xgboost.csv", sep = ",")
train = pd.read_csv("train_xgboost.csv", sep = ",")

categorical_features = ["IS_ALONE",
						"HAUL_TYPE",
						"DEVICE",
						"TRIP_TYPE",
						"COMPANY"]

numerical_features = ["FAMILY_SIZE",
					  "DISTANCE",
					  "TRIP_LEN_DAYS"]

train = pd.get_dummies(train, columns = categorical_features)
test = pd.get_dummies(test, columns = categorical_features)

train["EXTRA_BAGGAGE"] = LabelEncoder().fit_transform(train["EXTRA_BAGGAGE"])

dummy_categorical_features = ["IS_ALONE_ALONE",
							  "HAUL_TYPE_CONTINENTAL",
							  "HAUL_TYPE_DOMESTIC",
							  "HAUL_TYPE_INTERCONTINENTAL",
							  "DEVICE_SMARTPHONE",
							  "DEVICE_TABLET",
							  "DEVICE_COMPUTER",
							  "TRIP_TYPE_MULTI_DESTINATION",
							  "TRIP_TYPE_ONE_WAY",
							  "TRIP_TYPE_ROUND_TRIP",
							  "COMPANY_EDREAMS",
							  "COMPANY_GO_VOYAGE",
							  "COMPANY_OPODO"]

features = dummy_categorical_features + numerical_features

# Separation of features and outcome
X_train = train[list(features)].values
Y_train = train["EXTRA_BAGGAGE"].values
X_test = test[list(features)].values

# Train/validation split to compute F1 and AUC internally without holdout set
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train,Y_train,
																test_size=0.2,
																random_state=7)

# Rule-of-thumb XGBoost parameters
params = {'tree_method': 'gpu_hist',
          'updater' : 'grow_gpu',
          'n_jobs': -1
          }
n_estimators=400
max_depth=5
min_child_weight=5
learning_rate=0.1

clf = XGBClassifier(n_estimators=n_estimators,
					max_depth=max_depth,
					min_child_weight=min_child_weight,
					learning_rate=learning_rate,
					**params)
# clf = XGBClassifier(**params)
clf.fit(X_train, Y_train)

Y_test = clf.predict(X_test)
Y_test_proba = clf.predict_proba(X_test)

Y_validation_proba = clf.predict_proba(X_validation)

print("=====================  Model parameters  =====================")
print(clf)

print("=====================  Validation set  =======================")
print("AUROC = {}".format(roc_auc_score(Y_validation,
										clf.predict(X_validation))))

print("======================  Training set  ========================")
print("AUROC = {}".format(roc_auc_score(Y_train,
										clf.predict(X_train))))

print("==============================================================")

f = open("submission_xgboost.csv", 'w')
f.write("ID,EXTRA_BAGGAGE\n")
for i in range(len(Y_test)):
    # f.write("{},{}\n".format(i,Y_test[i]))
    f.write("{},{:6f}\n".format(i,Y_test_proba[i,0]))

# # plot feature importance
# plot_importance(clf)
# plt.show()

array = np.column_stack((Y_validation_proba[:,1], Y_validation))

x = array[np.where(array[:,1] == 1.)]
y = array[np.where(array[:,1] == 0.)]
sns.distplot(x[:,0], label='positives')
sns.distplot(y[:,0], label='negatives')
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.xlim(0, 1)
plt.legend(loc='upper right')
plt.show()

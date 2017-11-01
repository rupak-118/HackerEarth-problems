### HackerEarth ML Challenge for Beginners - Predict Customer Happiness

## Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re, nltk

## Reading the datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

## EDA
train.describe(include = ['O'])
test.describe(include = ['O'])

pd.value_counts(train.Browser_Used)
pd.value_counts(train.Device_Used)

pd.crosstab(train.Browser_Used, train.Is_Response)
pd.crosstab(train.Device_Used, train.Is_Response)

# Clubbing similar browsers to reduce levels of Browser_Used
orig_browser_list = ['Chrome', 'Edge', 'Firefox', 'Google Chrome', 'IE', 'Internet Explorer',
                     'InternetExplorer', 'Mozilla', 'Mozilla Firefox', 'Opera', 'Safari']
new_browser_list = ['Chrome', 'Edge', 'Firefox', 'Chrome', 'IE', 'IE', 'IE', 'Firefox', 'Firefox', 'Opera', 'Safari']
browser_map = pd.Series(data = new_browser_list, index = orig_browser_list)
train.Browser_Used = train.Browser_Used.map(browser_map)
test.Browser_Used = test.Browser_Used.map(browser_map)

# Label encoding the target variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['Is_Response'] = le.fit_transform(list(train['Is_Response'].values))

# A very naive CatBoost model
X = train.iloc[:,2:-1].values
y = train.Is_Response.values
X_test = test.iloc[:,2::].values

# Train-validation split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)

from catboost import CatBoostClassifier
model = CatBoostClassifier(iterations = 50, learning_rate = 0.1, depth = 2, l2_leaf_reg = 10, loss_function = 'Logloss',
                           use_best_model = True)
cat_features = [0,1]
model.fit(X_train, y_train, cat_features, eval_set = (X_val, y_val))

model.score(X_val, y_val)
preds_val = model.predict(X_val)
from sklearn.metrics import confusion_matrix
cm_val = confusion_matrix(y_val, preds_val)

preds = model.predict(X_test)








## Submitting predictions
subm = pd.DataFrame({'User_ID': test.User_ID.values  , 'Is_Response': preds})
subm['Is_Response'] = ['happy' if x == 0 else 'not_happy' for x in subm['Is_Response']]
subm = subm[['User_ID','Is_Response']]
subm.to_csv('sub01.csv', index = False)













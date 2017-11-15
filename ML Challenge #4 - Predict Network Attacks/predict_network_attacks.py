### HackerEarth ML Challenge #4 - Predict Network Attacks 

## Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import re,tqdm

## Reading datasets
train = pd.read_csv('train_data.csv')
test = pd.read_csv('test_data.csv')

train.head()
train.target.value_counts(normalize = True)
''' 0 is the majority class. A little bit of re-sampling techniques can be tried '''

train.describe()
train.describe(include = ['O']) # only connection_id

train_na = train.isnull().sum()/train.shape[0]
test_na = test.isnull().sum()/test.shape[0]
''' No missing values in either of the datasets '''

sns.distplot(train.cont_2, hist = False)


## 1. Try different algorithms - LightGBM, CatBoost, XGB, SVM, , DecisionTree/RF, ANNs(standardized) etc.
## 2. Feature engg. - Find important features and try to create new features out of it
## 3. Resampling to have almost equal target class distributions + hyper-parameter tuned algos from point 1
## 4. Individual feature exploration - distributions, transformations (if reqd.) etc.
## 5. Ensembles

'''
Exploration of various features tell us that classes 0 and 2 have very similar feature values
and are difficult to segregate, whereas class 1 examples are easily separable from rest of
the classes. Hence, the focus would be on getting perfect classification for class 1 and
somehow find a way to separate classes 0 and 2 (maybe use id as a feature)
'''

''' 
The categorical features are already label encoded. Either OHE can be done or the model
can be run as-is
'''

# Checking column types
for c in train.columns:
    print(train[c].dtype)
''' All the cont_x features are of float64 type and all the cat_x features are of int64 type
'''

# Modifying connection_id
def extract_num(s):
    return float(s.split(sep = '_')[1])

train['connection_id'] = train.connection_id.apply(extract_num).map(int)


# Building feature-set
X = train.iloc[:,1:-1].values # without connection_id
y = train['target'].values
X_test = test.iloc[:,1::].values

# Train-validation split
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val = train_test_split(X, y, test_size = 0.2, stratify = y)


## Model 1 : CatBoost
from catboost import CatBoostClassifier
cat_features = list(range(18,41))
model = CatBoostClassifier(iterations = 500, learning_rate = 0.2, depth = 8, l2_leaf_reg = 20,
                           loss_function = 'MultiClass', use_best_model = True)
model.fit(X_train, y_train, cat_features, eval_set = (X_val, y_val))

preds_val = model.predict(X_val)
preds_cb = model.predict(X_test)[:,0]
prob_val = model.predict_proba(X_val)
feat_imp_CatBoost = model.feature_importance_

# Exploring predictions on validation set
from sklearn.metrics import confusion_matrix
confusion_matrix(y_val, preds_val)
''' 
Only two classes are being predicted. Class 2 is being masked by class 0. Expected, since 
both classes have elements having similar features. 
'''

## Model 2 : RF - a simpler model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 501, criterion = "gini", max_depth = 6, 
                               max_features = "auto", n_jobs = -1)
model.fit(X_train, y_train)

preds_val = model.predict(X_val)
prob_val = model.predict_proba(X_val)
preds_rf = model.predict(X_test)


## Model 3 : LightGBM
import lightgbm as lgb
train_data = lgb.Dataset(X_train, label = y_train)
params = {'num_leaves':70, 'objective': 'multiclass', 'max_depth':6, 'learning_rate':0.1, 
         'max_bin':111, 'feature_fraction':0.9, 'bagging_fraction':0.7, 'lambda_l1': 10,
         'lambda_l2':1, 'nthread' : -1, 'metric':'multi_logloss', 'num_class': 3}
model = lgb.train(params, train_data, num_boost_round = 100)

# Check validation score
preds_val = np.argmax(model.predict(X_val), 1) 
confusion_matrix(y_val, preds_val)


## Model 4 : XGBoost
from xgboost import XGBClassifier
model = XGBClassifier(max_depth = 6, learning_rate = 0.01,
                            n_estimators = 300, objective = "multi:softmax", 
                            gamma = 0, base_score = 0.5, reg_lambda = 1, subsample = 0.6,
                            colsample_bytree = 0.8, num_class = 3)

model.fit(X_train, y_train, eval_metric = "error")
feat_imp = model.feature_importances_





## Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators' : [51, 100, 201, 300, 500]}
             ]

grid_search = GridSearchCV(estimator = model, 
                           param_grid = parameters,
                           scoring = None,
                           cv = 5, n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_metric = grid_search.best_score_
best_params = grid_search.best_params_
grid_search.grid_scores_ # See all scores



## Trying to segregate classes 0 and 2
trn_0_2 = train[train.target != 1]
X_0_2 = trn_0_2.iloc[:,:-1].values
y_0_2 = trn_0_2['target'].values

X_trn_0_2,X_val_0_2,y_trn_0_2,y_val_0_2 = train_test_split(X_0_2, y_0_2, test_size = 0.2)

# XGB Classifier
model = XGBClassifier(max_depth = 12, learning_rate = 0.1,
                            n_estimators = 21, objective = "binary:logistic", 
                            gamma = 0, base_score = 0.5, reg_lambda = 0, subsample = 0.7,
                            colsample_bytree = 0.8)

model.fit(X_trn_0_2, y_trn_0_2, eval_metric = "error")
feat_imp = model.feature_importances_

preds_val_0_2 = model.predict(X_val_0_2)
confusion_matrix(y_val_0_2,preds_val_0_2)
''' 
Still unable to segregate classes 0 and 2. Although connection_id turns out to be the
most important feature. A little segregation is observed on overfitting the dataset by
using appropriate model parameters. But that will mostly, decrease the overall accuracy.
'''

# Kernel SVM
from sklearn.svm import SVC
model = SVC(kernel = 'rbf', C = 0.1, probability = False)
model.fit(X_trn_0_2, y_trn_0_2)

preds_val_0_2 = model.predict(X_val_0_2)
confusion_matrix(y_val_0_2,preds_val_0_2)
''' Kernel SVMs have very high time complexity. Hence, should only be tried with very few features, along
    with feature scaling and subsamplng dataset (maybe use, Bagging Classifier)
'''


# Submitting predictions
subm = pd.DataFrame({'connection_id':test['connection_id'].values, 'target': preds_cb})
subm.to_csv('sub07.csv', index=False)

















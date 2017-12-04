### HackerEarth ML Challenge #4 - Predict Network Attacks 

## Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import re,tqdm
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, GridSearchCV

## Reading datasets
train = pd.read_csv('train_data.csv')
test = pd.read_csv('test_data.csv')

## EDA
train.head()
train.target.value_counts(normalize = True)
''' 0 is the majority class. A little bit of re-sampling techniques can be tried '''

train.describe()
train.describe(include = ['O']) # only connection_id

train_na = train.isnull().sum()/train.shape[0]
test_na = test.isnull().sum()/test.shape[0]
''' No missing values in either of the datasets '''

sns.distplot(train.cont_2, hist = False)

## Checking feature cardinality , specially for categorical variables
feat_cardinality = train.apply(pd.Series.nunique)
feat_cardinality_test = test.apply(pd.Series.nunique)
''' High cardinality categorical variables will undergo some sort of target encoding, rest OHE/as-it-is '''

# Removing cat_17 feature as it has only one value
train.drop('cat_17', axis = 1, inplace = True)
test.drop('cat_17', axis = 1, inplace = True)

## Checking correlations between features
cont_features = list(train.columns[train.columns.str.startswith('cont_')])
cat_features = list(train.columns[train.columns.str.startswith('cat_')])
feat_corr = train[cont_features].corr()
plt.figure(figsize = (16,10))
sns.heatmap(feat_corr)
''' Quite a few variables are correlated '''


'''
Exploration of various features tell us that classes 0 and 2 have very similar feature values
and are difficult to segregate, whereas class 1 examples are easily separable from rest of
the classes. Hence, the focus would be on getting perfect classification for class 1 and
somehow find a way to separate classes 0 and 2 (maybe, use id as a feature).

This version of the script only focuses on multi-classification without putting extra effort
in segregating classes 0 and 2. Approaches are listed below.
'''

### 1. Data pre-processing : Decorrelate features + categorical variable encoding
### 2. Check for co-variate shift : Drop drifting variables
### 3. Feature engineering with xgbfi/xgbfir to derive higher order interaction features to boost model accuracy
### 4. Different tree and GBDT models - XGBoost, LightGBM, RF, CatBoost etc.
### 5. Cross-validation setup (KFold/StratifiedKFold)
### 6. Ensembles - Stacking of various tree-based algorithms 


## Encoding of categorical variables



## Decorrelating continuous features with PCA




## Building feature-sets
X = train.iloc[:,1:-1].values # without connection_id
y = train['target'].values
X_test = test.iloc[:,1::].values


## Model 1 : CatBoost
from catboost import CatBoostClassifier
model = CatBoostClassifier(iterations = 500, learning_rate = 0.2, depth = 8, l2_leaf_reg = 20,
                           loss_function = 'MultiClass', use_best_model = True)
cat_feat = list(range(18,40))
# Train-validation split
from sklearn.model_selection import train_test_split
X_train,X_val,y_train,y_val = train_test_split(X, y, test_size = 0.2, stratify = y)

model.fit(X_train, y_train, cat_feat, eval_set = (X_val, y_val))
preds = model.predict(X_test)[:,0]

## Model 2 : Simple RF model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 501, criterion = "gini", max_depth = 6, 
                               max_features = "auto", n_jobs = -1)
model.fit(X, y)


## Model 3 : LightGBM
import lightgbm as lgb
train_data = lgb.Dataset(X, label = y)
params = {'num_leaves':70, 'objective': 'multiclass', 'max_depth':6, 'learning_rate':0.1, 
         'max_bin':111, 'feature_fraction':0.9, 'bagging_fraction':0.7, 'lambda_l1': 10,
         'lambda_l2':1, 'nthread' : -1, 'metric':'multi_logloss', 'num_class': 3}
model = lgb.train(params, train_data, num_boost_round = 100)


## Model 4 : XGBoost
from xgboost import XGBClassifier
model = XGBClassifier(max_depth = 6, learning_rate = 0.01,
                            n_estimators = 300, objective = "multi:softmax", 
                            gamma = 0, base_score = 0.5, reg_lambda = 1, subsample = 0.6,
                            colsample_bytree = 0.8)

model.fit(X, y, eval_metric = "error", n_jobs = -1)
feat_imp = model.feature_importances_


# Using the cross_val_score below to directly calculate avg. eval metric for n-fold CV
scores = cross_val_score(model, X, y, cv = 5, scoring = 'accuracy', n_jobs = 8)


## Applying Grid Search to find the best model and the best parameters
parameters = [{'n_estimators' : [51, 100, 301, 500],
               'max_depth' : [3,6,8,9],
               'learning_rate' = [0.01, 0.1, 0.5]}
             ]

grid_search = GridSearchCV(estimator = model, 
                           param_grid = parameters,
                           scoring = None,
                           cv = 5, n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_metric = grid_search.best_score_
best_params = grid_search.best_params_
grid_search.grid_scores_ # See all scores


## Building structure for Ensemble modelling (Stacking)
class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, X_test):
        X = np.array(X)
        y = np.array(y)
        X_test = np.array(X_test)

        folds = list(StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state = 42).split(X, y))

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((X_test.shape[0], len(self.base_models)))
        
        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((X_test.shape[0], self.n_splits))

            for j, (train_idx, val_idx) in enumerate(folds):
                X_train, X_val, y_train, y_val = X[train_idx], X[val_idx], y[train_idx], y[val_idx]
                print ("Fit %s fold %d" % (str(clf).split('(')[0], j+1))
                clf.fit(X_train, y_train)
                
                y_pred = clf.predict(X_val)
                S_train[val_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(X_test)
            S_test[:, i] = S_test_i.mean(axis=1)

        results = cross_val_score(self.stacker, S_train, y, cv=5, scoring='accuracy')
        print("Stacker score: %.5f" % (results.mean()))

        self.stacker.fit(S_train, y)
        res = self.stacker.predict(S_test)
        return res

## Different models used for the ensemble
lgb_model = LGBMClassifier(learning_rate = 0.01, n_estimators = 650, max_bin = 81, subsample = 0.8, feature_fraction = 0.9,
                           subsample_freq = 0, max_depth = 8, bagging_freq = 1, random_state = 42)

xgb1_model = XGBClassifier(max_depth = 6, learning_rate = 0.1,
                            n_estimators = 501, objective = "multi:softmax", 
                            gamma = 0, base_score = 0.5, reg_lambda = 1, subsample = 0.75,
                            colsample_bytree = 0.8)

xgb2_model = XGBClassifier(max_depth = 9, learning_rate = 0.01,
                            n_estimators = 780, objective = "multi:softmax", 
                            gamma = 0, base_score = 0.5, reg_lambda = 5, subsample = 0.75,
                            colsample_bytree = 0.8)

log_model = LogisticRegression()
       
stack = Ensemble(n_splits=4,
        stacker = log_model,
        base_models = (lgb_model, xgb1_model, xgb2_model))        
        
ensemble_pred = stack.fit_predict(X, y, X_test)        



## Submitting predictions
subm = pd.DataFrame({'connection_id':test['connection_id'].values, 'target': ensemble_pred})
subm.to_csv('sub09.csv', index=False)











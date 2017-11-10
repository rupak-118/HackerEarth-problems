### HackerEarth ML Challenge #4 - Predict Network Attacks 

## Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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



# Building feature-set
X = train.iloc[:,1:-1].values
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
''' Only two classes are being predicted. Class 2 is being masked by class 0. Maybe both 
    classes have elements having similar features. Looking at each feature individually
    might give some information and/or resampling class 2 might help '''


# Trying a simpler algorithm to see if all classes are predicted properly
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 501, criterion = "gini", max_depth = 6, 
                               max_features = "auto", n_jobs = -1)
model.fit(X_train, y_train)

preds_val = model.predict(X_val)
prob_val = model.predict_proba(X_val)
preds_rf = model.predict(X_test)
''' Still the same problem as CatBoost algorithm, which means it's not about the algorithm,
    but the data, feature distribution, re-sampling etc. which needs to be taken care of '''
''' Re-sampling won't solve this. Tried using class_weights in RF model '''




# Submitting predictions
subm = pd.DataFrame({'connection_id':test['connection_id'].values, 'target': preds_cb})
subm.to_csv('sub07.csv', index=False)



## Model 2 : Light GBM
# Create a LightGBM-compatible metric from Gini
def gini_xgb(preds, train_data):
    labels = train_data.get_label()
    gini_score = gini_normalized(labels, preds)
    return [('gini', gini_score)]


# Light GBM
import lightgbm as lgb
train_data = lgb.Dataset(X_train, label = y_train)
params = {'num_leaves':151, 'objective': 'binary', 'max_depth':9, 'learning_rate':0.1, 
         'max_bin':222, 'feature_fraction':0.9, 'bagging_fraction':0.7, 'lambda_l1': 10,
         'lambda_l2':1}
params['metric'] = ['auc','binary_logloss']
#params['metric'] = gini_xgb
num_rounds = 150
model_lgb = lgb.train(params, train_data, num_rounds)

# Check validation score
pred_val = model_lgb.predict(X_val)
pred_trn = model_lgb.predict(X_train)













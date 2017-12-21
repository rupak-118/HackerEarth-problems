### BrainWaves ML Challenge (HackerEarth) - Predict Annual returns

## Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV

## Importing datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

## EDA

train.describe()
train.describe(include = ['O'])
test.describe()

sns.distplot(train['return'], hist = False)
# check missing values
train_na = train.isnull().sum()/train.shape[0]
test_na = test.isnull().sum()/test.shape[0]

# Check feature cardinality
feat_cardinality = train.apply(pd.Series.nunique)
feat_cardinality_test = test.apply(pd.Series.nunique)

# Correlation b/n numerical variables
corr_vars = ['sold', 'bought', 'euribor_rate', 'libor_rate', 'return']
corr_df = train[corr_vars].corr()

'''
1. 3 date variables, 3 boolean variables, 2 ID variables, 5 categorical variables and 4 numerical variables
2. Missing values present in some columns
3. Categorical variables do not have too many levels. A simple OHE/dummy encoding would do
4. 'sold' and 'bought' are completely correlated (both having almost the same value)
5. 'return' variable is significantly correlated with euribor and libor rate
6. 'euribor_rate' and 'currency' together seem to be highly correlated to 'libor_rate',
    which means the missing values in 'libor_rate' can be removed using euribor_rate and currency in a kNN model
'''

## Missing value imputation
# Deleting train set rows where sold and bought are empty
train = train[train.sold.isnull() == False]

# Filling NaNs
train.hedge_value.fillna('MISSING', inplace = True)
train.indicator_code.fillna(False, inplace = True)
train.status.fillna(False, inplace = True)
train.desk_id.fillna('MISSING', inplace = True)

test.hedge_value.fillna('MISSING', inplace = True)
test.indicator_code.fillna(False, inplace = True)
test.status.fillna(False, inplace = True)
test.desk_id.fillna('MISSING', inplace = True)

# Converting True/False to numeric values
true_false_map = {False: 0, True: 1, 'MISSING': 2}
train['hedge_value'] = train.hedge_value.map(true_false_map)
train['indicator_code'] = train.indicator_code.map(true_false_map)
train['status'] = train.status.map(true_false_map)

test['hedge_value'] = test.hedge_value.map(true_false_map)
test['indicator_code'] = test.indicator_code.map(true_false_map)
test['status'] = test.status.map(true_false_map)

## Categorical variable encoding
# Encoding for currency based on their value
currency_map = {'JPY': 0, 'USD': 1, 'CHF': 2, 'EUR': 3, 'GBP': 4}
train['currency'] = train.currency.map(currency_map)
test['currency'] = test.currency.map(currency_map)
# Label Encoding for remaining categorical variables
cols_to_encode = ['office_id', 'desk_id', 'pf_category', 'country_code', 'type']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df = pd.concat([train,test]) #combining train and test datasets
df = df[train.columns] # re-ordering according to train columns 
for c in cols_to_encode:
    df[c] = le.fit_transform(df[c].values)
# separate train and test after label encoding
train = df.iloc[0:train.shape[0],:]
test = df.iloc[train.shape[0]::,:-1]


## Removing missing values from libor_rate using kNN
features = ['euribor_rate', 'currency']
libor_train = df[df.libor_rate.isnull() == False][features]
libor_y = df[df.libor_rate.isnull() == False]['libor_rate']
libor_test = df[df.libor_rate.isnull() == True][features]

from sklearn.neighbors import KNeighborsRegressor
kNN_reg = KNeighborsRegressor(n_neighbors = 3)
kNN_reg.fit(libor_train,libor_y)
libor_test_y = kNN_reg.predict(libor_test)
''' The predicted missing value is luckily, just a single value which can be broadcasted
    across train and test '''

# Filling missing_value in libor_rate
train.libor_rate.fillna(0.16196925, inplace = True)
test.libor_rate.fillna(0.16196925, inplace = True)


## Changing date columns to suitable formats and extracting features



## Creating new features
train['profit_loss'] = train['sold'] - train['bought']
test['profit_loss'] = test['sold'] - test['bought']

## Building regression models
# Defining features
reg_features = ['office_id', 'pf_category', 'country_code', 'euribor_rate', 'currency',
                'libor_rate', 'bought', 'sold', 'profit_loss', 'indicator_code', 'type',
                'hedge_value', 'status']
X = train[reg_features].values
y = train['return'].values
X_test = test[reg_features].values

# Feature Scaling and normalization of only the large valued columns
''' A simple scaling would also have sufficed '''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X[:,6:9] = sc.fit_transform(X[:,6:9])
X_test[:,6:9] = sc.transform(X_test[:,6:9])
X = sc.fit_transform(X)
X_test = sc.transform(X_test)

''' Alternately, only scaling the three columns '''
X[:,6:9] = X[:,6:9]/1000000
X_test[:,6:9] = X_test[:,6:9]/1000000


# Model 1: Support Vector Regression (SVR)
from sklearn.svm import SVR
''' Will try linear SVM, kernel SVM as well as polynomial SVM regressors '''
model = SVR(kernel = 'rbf', gamma = 'auto', C = 10)
model.fit(X, y)

fold_num = 1
cv_scores = []
skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
for train_idx, val_idx in skf.split(X,y):
    print("Fitting fold %d" %fold_num)
    model.fit(X[train_idx], y[train_idx])
    score = r2_score(y[val_idx],model.predict(X[val_idx]))
    cv_scores.append(score)
    print("Eval. score (R2-score) for fold {} = {}\n".format(fold_num,score))
    fold_num += 1

# Predicting on test set
preds = model.predict(X_test)

''' SVR doesn't seem to be working well. For any possible combination, it's giving negative scores '''

# Model 2 : Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators = 300, criterion = "mse", max_depth = 12)
model.fit(X,y)
# cross-validation
fold_num = 1
cv_scores = []
skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 102)
for train_idx, val_idx in skf.split(X,y):
    print("Fitting fold %d" %fold_num)
    model.fit(X[train_idx], y[train_idx])
    score = r2_score(y[val_idx],model.predict(X[val_idx]))
    cv_scores.append(score)
    print("Eval. score (R2-score) for fold {} = {}\n".format(fold_num,score))
    fold_num += 1

# Predicting on test set
preds = model.predict(X_test)





## Applying Grid Search to find the best model and the best parameters
parameters = [{'n_estimators' : [120, 343],
               'max_depth' : [6, 10],
               'min_samples_split' : [2],
               'min_samples_leaf' : [1]}
             ]

grid_search = GridSearchCV(estimator = model, 
                           param_grid = parameters,
                           scoring = 'r2',
                           cv = 5, n_jobs = 1)
grid_search = grid_search.fit(X, y)
best_metric = grid_search.best_score_
best_params = grid_search.best_params_
grid_search.grid_scores_ # See all scores








## Submitting predictions
subm = pd.DataFrame({'portfolio_id':test['portfolio_id'].values, 'return': preds}, 
                     columns = ['portfolio_id','return'])
subm.to_csv('sub02.csv', index=False)


















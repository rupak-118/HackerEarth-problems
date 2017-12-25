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
train['start_date'] = pd.to_datetime(train['start_date'], format = '%Y%m%d')
train['creation_date'] = pd.to_datetime(train['creation_date'], format = '%Y%m%d')
train['sell_date'] = pd.to_datetime(train['sell_date'], format = '%Y%m%d')

train['start_year'] = train['start_date'].apply(lambda x: x.year)
train['start_month'] = train['start_date'].apply(lambda x: x.month)
train['start_day'] = train['start_date'].apply(lambda x: x.day)
train['start_weekday'] = train['start_date'].apply(lambda x: x.weekday())
train['creation_year'] = train['creation_date'].apply(lambda x: x.year)
train['creation_month'] = train['creation_date'].apply(lambda x: x.month)
train['creation_day'] = train['creation_date'].apply(lambda x: x.day)
train['creation_weekday'] = train['creation_date'].apply(lambda x: x.weekday())
train['sell_year'] = train['sell_date'].apply(lambda x: x.year)
train['sell_month'] = train['sell_date'].apply(lambda x: x.month)
train['sell_day'] = train['sell_date'].apply(lambda x: x.day)
train['sell_weekday'] = train['sell_date'].apply(lambda x: x.weekday())


test['start_date'] = pd.to_datetime(test['start_date'], format = '%Y%m%d')
test['creation_date'] = pd.to_datetime(test['creation_date'], format = '%Y%m%d')
test['sell_date'] = pd.to_datetime(test['sell_date'], format = '%Y%m%d')

test['start_year'] = test['start_date'].apply(lambda x: x.year)
test['start_month'] = test['start_date'].apply(lambda x: x.month)
test['start_day'] = test['start_date'].apply(lambda x: x.day)
test['start_weekday'] = test['start_date'].apply(lambda x: x.weekday())
test['creation_year'] = test['creation_date'].apply(lambda x: x.year)
test['creation_month'] = test['creation_date'].apply(lambda x: x.month)
test['creation_day'] = test['creation_date'].apply(lambda x: x.day)
test['creation_weekday'] = test['creation_date'].apply(lambda x: x.weekday())
test['sell_year'] = test['sell_date'].apply(lambda x: x.year)
test['sell_month'] = test['sell_date'].apply(lambda x: x.month)
test['sell_day'] = test['sell_date'].apply(lambda x: x.day)
test['sell_weekday'] = test['sell_date'].apply(lambda x: x.weekday())

''' All the year fields begin from 2004. Hence, using it as reference and subtracting 2014
    from each of these fields. This helps in bringing the column values to the same scale as others
'''
train['start_year'] = train['start_year'] - 2004
train['creation_year'] = train['creation_year'] - 2004
train['sell_year'] = train['sell_year'] - 2004
test['start_year'] = test['start_year'] - 2004
test['creation_year'] = test['creation_year'] - 2004
test['sell_year'] = test['sell_year'] - 2004


## Creating other new features
''' 'sold' and 'bought' don't have any negative or zero values. Hence log10() can be used
     instead of log1p()
'''
train['log_sold'] = np.log10(train['sold'])
train['log_bought'] = np.log10(train['bought'])
test['log_sold'] = np.log10(test['sold'])
test['log_bought'] = np.log10(test['bought'])

train['profit_loss'] = train['sold'] - train['bought']
test['profit_loss'] = test['sold'] - test['bought']

# Creating features using difference of dates
train['sell-start'] = (train.sell_date - train.start_date).apply(lambda x: x.days)
train['creation-start'] = (train.creation_date - train.start_date).apply(lambda x: x.days)
train['sell-creation'] = (train.sell_date - train.creation_date).apply(lambda x: x.days)
test['sell-start'] = (test.sell_date - test.start_date).apply(lambda x: x.days)
test['creation-start'] = (test.creation_date - test.start_date).apply(lambda x: x.days)
test['sell-creation'] = (test.sell_date - test.creation_date).apply(lambda x: x.days)

## Building regression models
# Defining features
reg_features = ['office_id', 'pf_category', 'country_code', 'euribor_rate', 'currency',
                'libor_rate', 'log_bought', 'profit_loss', 'desk_id']
date_features = ['start_month', 'start_day', 'start_weekday', 'sell_month', 'sell_day', 
                 'sell_weekday', 'sell-start', 'sell-creation', 'creation-start']
all_features = reg_features + date_features

X = train[all_features].values
y = train['return'].values
X_test = test[all_features].values


# Scaling the 'profit_loss' column
X[:,7] = X[:,7]/1000000
X_test[:,7] = X_test[:,7]/1000000


# Model 1: Support Vector Regression (SVR)
from sklearn.svm import SVR
''' Will try linear SVM, kernel SVM as well as polynomial SVM regressors '''
model = SVR(kernel = 'rbf', gamma = 'auto', C = 10)
model.fit(X, y)

fold_num = 1
cv_scores = []
kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
for train_idx, val_idx in kf.split(X,y):
    print("Fitting fold %d" %fold_num)
    model.fit(X[train_idx], y[train_idx])
    score = r2_score(y[val_idx],model.predict(X[val_idx]))
    cv_scores.append(score)
    print("Eval. score (R2-score) for fold {} = {}\n".format(fold_num,score))
    fold_num += 1

# Predicting on test set
preds = model.predict(X_test)

''' SVR doesn't seem to be working well. For any possible combination, it's giving negative scores '''

# Model 2 : XGB Regression
from xgboost import XGBRegressor
model = XGBRegressor(max_depth = 12, learning_rate = 0.05, n_estimators = 501, reg_alpha = 0,
                     reg_lambda = 10, subsample = 0.8)
model.fit(X,y)
# cross-validation
fold_num = 1
cv_scores = []
kf = KFold(n_splits = 5, shuffle = True, random_state = 102)
for train_idx, val_idx in kf.split(X,y):
    print("Fitting fold %d" %fold_num)
    model.fit(X[train_idx], y[train_idx], eval_metric = "rmse")
    score = r2_score(y[val_idx],model.predict(X[val_idx]))
    cv_scores.append(score)
    print("Eval. score (R2-score) for fold {} = {}\n".format(fold_num,score))
    fold_num += 1

print("Mean CV score = {}; Std. dev. CV score = {}\n".format(np.mean(cv_scores), np.std(cv_scores)))
feat_imp = pd.DataFrame(data = model.feature_importances_, index = top_10_features)


## Using xgbfir to learn more about feature interactions and create new useful features
import xgbfir
# saving to file with proper feature names
xgbfir.saveXgbFI(model, feature_names = all_features, OutputXlsxFile='predict_returns_FI.xlsx')

# Creating new features based on XGBFI file
train['country_desk_id'] = train['country_code']*10000 + train['desk_id']
train['pr_loss_maxibor'] = train[['euribor_rate', 'libor_rate']].apply(max, axis = 1) * train['profit_loss']
train['pr_loss_euribor'] = train['profit_loss'] * train['euribor_rate']
train['pr_loss_libor'] = train['profit_loss'] * train['libor_rate']
train['currency_euribor_pr_loss'] = train['currency'] * train['pr_loss_euribor']

test['country_desk_id'] = test['country_code']*10000 + test['desk_id']
test['pr_loss_maxibor'] = test[['euribor_rate', 'libor_rate']].apply(max, axis = 1) * test['profit_loss']
test['pr_loss_euribor'] = test['profit_loss'] * test['euribor_rate']
test['pr_loss_libor'] = test['profit_loss'] * test['libor_rate']
test['currency_euribor_pr_loss'] = test['currency'] * test['pr_loss_euribor']


## Redefining features
reg_features = ['office_id', 'pf_category', 'country_code', 'euribor_rate', 'currency',
                'libor_rate', 'log_bought', 'profit_loss', 'desk_id']
date_features = ['start_month', 'start_day', 'start_weekday', 'sell_month', 'sell_day', 
                 'sell_weekday', 'sell-start', 'sell-creation', 'creation-start']
interaction_features = ['country_desk_id', 'pr_loss_maxibor', 'pr_loss_euribor',
                        'pr_loss_libor', 'currency_euribor_pr_loss']
all_features = reg_features + date_features + interaction_features

top_10_features = ['log_bought', 'euribor_rate', 'desk_id', 'libor_rate', 'country_desk_id',
                   'profit_loss', 'currency_euribor_pr_loss', 'sell-start', 
                   'pr_loss_maxibor', 'pr_loss_euribor']

X = train[top_10_features].values
y = train['return'].values
X_test = test[top_10_features].values

# Scaling the relevant columns
X[:,5] = X[:,5]/1000000
X_test[:,5] = X_test[:,5]/1000000

''' We re-run the XGB Regressor model now '''


## Model 3 : LightGBM Regression
from lightgbm import LGBMRegressor
model = LGBMRegressor(num_leaves = 22, max_depth = 10, learning_rate = 0.1, max_bin = 45,
                      n_estimators = 4510, subsample = 1, reg_alpha = 0, reg_lambda = 15)
model.fit(X, y)
# cross-validation
fold_num = 1
cv_scores = []
kf = KFold(n_splits = 5, shuffle = True, random_state = 51)
for train_idx, val_idx in kf.split(X,y):
    print("Fitting fold %d" %fold_num)
    model.fit(X[train_idx], y[train_idx], eval_metric = "rmse")
    score = r2_score(y[val_idx],model.predict(X[val_idx]))
    cv_scores.append(score)
    print("Eval. score (R2-score) for fold {} = {}\n".format(fold_num,score))
    fold_num += 1

print("Mean CV score = {}; Std. dev. CV score = {}\n".format(np.mean(cv_scores), np.std(cv_scores)))


# Predicting on test set
preds = model.predict(X_test)
# Averaging predictions; preds_10 - Prediction using top 10 features, preds_all - using all features
preds = (preds_all + preds_10)/2 
 
sns.distplot(preds, hist = False)


#del X,y,X_test,model


## Applying Grid Search to find the best model and the best parameters
parameters = [{'n_estimators' : [2200, 2750, 3145, 4510],
               'max_depth' : [10],
               'learning_rate' : [0.1],
               'num_leaves' : [22],
               'max_bin' : [45],
               'reg_alpha' : [0],
               'reg_lambda' : [15],
               'subsample' : [1]}
             ]

#parameters = [{'n_estimators' : [650],
#               'max_depth' : [15, 18, 20],
#               'learning_rate' : [0.05],
#               'reg_alpha' : [0],
#               'reg_lambda' : [10],
#               'subsample' : [0.8]}
#             ]

grid_search = GridSearchCV(estimator = model, 
                           param_grid = parameters,
                           scoring = 'r2',
                           cv = 5, n_jobs = 1)
grid_search = grid_search.fit(X, y)
best_metric = grid_search.best_score_
best_params = grid_search.best_params_
grid_search.grid_scores_ # See all scores


## Building structure for ensemble models
class Ensemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, X_test):
        X = np.array(X)
        y = np.array(y)
        X_test = np.array(X_test)

        folds = list(KFold(n_splits=self.n_splits, shuffle=True, random_state = 102).split(X, y))

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

        results = cross_val_score(self.stacker, S_train, y, cv=5, scoring='r2')
        print("Stacker score: %.5f" % (results.mean()))

        self.stacker.fit(S_train, y)
        res = self.stacker.predict(S_test)
        return res


## Model and their parameters to be used for the stacked ensemble
xgb1 = XGBRegressor(max_depth = 12, learning_rate = 0.05, n_estimators = 501, reg_alpha = 0,
                     reg_lambda = 10, subsample = 0.8)

xgb2 = XGBRegressor(max_depth = 10, learning_rate = 0.05, n_estimators = 650, reg_alpha = 0,
                     reg_lambda = 5, subsample = 0.8)

xgb3 = XGBRegressor(max_depth = 15, learning_rate = 0.05, n_estimators = 501, reg_alpha = 0,
                     reg_lambda = 10, subsample = 0.8)

lgbm1 = LGBMRegressor(num_leaves = 22, max_depth = 10, learning_rate = 0.05, max_bin = 45,
                      n_estimators = 650, subsample = 0.8, reg_alpha = 0, reg_lambda = 5)

lgbm2 = LGBMRegressor(num_leaves = 22, max_depth = 10, learning_rate = 0.1, max_bin = 45,
                      n_estimators = 4510, subsample = 1, reg_alpha = 0, reg_lambda = 15)


## Layer 2 of stacked ensemble
from sklearn.linear_model import LinearRegression
linear_model = LinearRegression(normalize = False)
       
stack = Ensemble(n_splits=5,
        stacker = linear_model,
        base_models = (xgb1, xgb2, xgb3, lgbm1, lgbm2))        
        
ensemble_pred = stack.fit_predict(X, y, X_test)  


## Submitting predictions
subm = pd.DataFrame({'portfolio_id':test['portfolio_id'].values, 'return': ensemble_pred}, 
                     columns = ['portfolio_id','return'])
subm.to_csv('sub25.csv', index=False)


















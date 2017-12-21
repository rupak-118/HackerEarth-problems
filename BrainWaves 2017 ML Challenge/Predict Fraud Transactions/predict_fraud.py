### BrainWaves ML Challenge (HackerEarth) - Predict Fraudulent Transactions

## Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

## Importing datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


## EDA
train.describe()
train.describe(include = ['O'])
test.describe()
# check missing values
train_na = train.isnull().sum()/train.shape[0]
test_na = test.isnull().sum()/test.shape[0]

# check class imbalance ratio
pd.value_counts(train.target, normalize = True)

# separate numerical and categorical variables
num_vars = list(train.columns[train.columns.str.startswith('num_')])
cat_vars = list(train.columns[train.columns.str.startswith('cat_')])

# Check feature cardinality for categorical variables
feat_cardinality = train[cat_vars].apply(pd.Series.nunique)
feat_cardinality_test = test[cat_vars].apply(pd.Series.nunique)

# Check correlations between numerical features and the target
corr_vars = np.append(num_vars, 'target')
num_feat_corr = train[corr_vars].corr()
#num_feat_corr = train[num_vars].corr()
plt.figure(figsize = (8,5))
sns.heatmap(num_feat_corr)


''' 
1. Numerical features have similar distributions in train and test 
2. Imbalanced classification problem - 10% chance of a fraudulent transaction
3. Missing values in few categorical columns in train and test
4. Some categorical variables have very high cardinality, while some have single cardinality
5. num_var_1 and num_var_6 are highly correlated. Other numerical variables are moderately correlated to not correlated

'''

# Dropping single-cardinality features as they do not add any valuable information
cols_to_drop = ['cat_var_31', 'cat_var_35', 'cat_var_36', 'cat_var_37', 'cat_var_38', 
                'cat_var_40', 'cat_var_42'] 
train.drop(cols_to_drop, axis = 1, inplace = True)
test.drop(cols_to_drop, axis = 1, inplace = True)

# Dropping num_var_6 as it is fairly correlated with num_var_1 and num_var_7
train.drop('num_var_6', axis = 1, inplace = True)
test.drop('num_var_6', axis = 1, inplace = True)


## Missing value imputation
'''
cat_var_3 and cat_var_8 have a significant percentage of missing values in train and test.
Hence, adding a 'MISSING' category in these variables.
cat_var_1 and cat_var_6 have a very small percentage of missing values. Mode imputation would do.
'''
# mode imputation
train.cat_var_1.fillna(train.cat_var_1.mode()[0], inplace = True)
test.cat_var_1.fillna(train.cat_var_1.mode()[0], inplace = True)
test.cat_var_6.fillna(train.cat_var_6.mode()[0], inplace = True)

# Add 'MISSING' category
train.cat_var_3.fillna('MISSING', inplace = True)
train.cat_var_8.fillna('MISSING', inplace = True)
test.cat_var_3.fillna('MISSING', inplace = True)
test.cat_var_8.fillna('MISSING', inplace = True)


## Categorical variable encoding
''' Just using label encoding on all cat. features in the first iteration. Later on, we can 
    use target encoding + OHE based on the cardinality of the column
'''
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df = pd.concat([train,test]) #combining train and test datasets
#update cat_var list
cat_vars = list(train.columns[train.columns.str.startswith('cat_')])

for c in cat_vars:
    df[c] = le.fit_transform(df[c].values)

# separate train and test after label encoding
train = df.iloc[0:train.shape[0],:]
test = df.iloc[train.shape[0]::,:]

test.drop('target', axis = 1, inplace = True) #Dropping target variable filled with NaN from test set


## Creating feature-sets
X = train.iloc[:,:-2].values
X_test = test.iloc[:,:-1].values
y = train['target'].values

## Building models
# Model 1 : XGBoost + cross-validation
from xgboost import XGBClassifier
model = XGBClassifier(max_depth = 8, learning_rate = 0.1, n_estimators = 150,
                      objective = 'binary:logistic', gamma = 0, reg_lambda = 10, 
                      subsample = 0.8, reg_alpha = 1, colsample_bytree = 0.8, 
                      scale_pos_weight = 1)

fold_num = 1
cv_scores = []
skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
for train_idx, val_idx in skf.split(X,y):
    print("Fitting fold %d" %fold_num)
    model.fit(X[train_idx], y[train_idx], eval_metric = "auc")
    #np.append(feat_imp_list, model.feature_importances_, axis = 1)
    score = roc_auc_score(y[val_idx],model.predict_proba(X[val_idx])[:,1], average = 'macro')
    cv_scores.append(score)
    print("Eval. score (ROC-AUC) for fold {} = {}\n".format(fold_num,score))
    fold_num += 1

# or, we can also use cross_val_score function
cv_results = cross_val_score(model, X, y, cv = 5, scoring = 'roc_auc')
model.fit(X,y, eval_metric = "auc")

# checking on importance of features 
feat_imp = model.feature_importances_
''' Numerical variables 7,2 and 1 are important in that order '''   
# Predicting on test set
preds = model.predict_proba(X_test)[:,1]


# Model 2 : LightGBM





## Applying Grid Search to find the best model and the best parameters
parameters = [{'n_estimators' : [51, 150, 301, 500],
               'max_depth' : [8],
               'learning_rate' : [0.05, 0.1],
               'reg_lambda' : [1],
               'reg_alpha' : [0],
               'scale_pos_weight' : [1, 9]}
             ]

grid_search = GridSearchCV(estimator = model, 
                           param_grid = parameters,
                           scoring = 'roc_auc',
                           cv = 5, n_jobs = -1)
grid_search = grid_search.fit(X, y)
best_metric = grid_search.best_score_
best_params = grid_search.best_params_
grid_search.grid_scores_ # See all scores








## Submitting predictions
subm = pd.DataFrame({'transaction_id':test['transaction_id'].values, 'target': preds}, 
                     columns = ['transaction_id','target'])
subm.to_csv('sub05.csv', index=False)






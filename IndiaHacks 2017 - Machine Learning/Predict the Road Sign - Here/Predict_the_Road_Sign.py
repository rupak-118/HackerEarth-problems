# Predict the Road Sign - Here maps

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Importing the dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

## Previewing datasets
train.info()
print("-"*40)
test.info()
''' Findings - No null values exist in any column '''
# Print summary statistics of each column
train.describe() # --> Summary statistics of numerical features + Target
train.describe(include = ['O']) # --> Summary statistics of categorical features
test.describe()
test.describe(include = ['O'])

# Create numpy arrays for train and test datasets
X_train = train.iloc[:, 1:-1].values
y_train = train.iloc[:, -1].values
X_test = test.iloc[:,1::].values

# Encoding categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X_train[:, 0] = labelencoder_X.fit_transform(X_train[:, 0])
X_test[:, 0] = labelencoder_X.transform(X_test[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_test = onehotencoder.transform(X_test).toarray()
labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)
# Avoiding dummy variable trap
X_train = X_train[:, 1::]
X_test = X_test[:, 1::]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Model 1 : Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 501, criterion = "entropy", n_jobs = -1)
classifier.fit(X_train, y_train)


# Model 2 : SVM Classification
'''Feature scaling is advisable before using SVM '''
from sklearn.svm import SVC
classifier2 = SVC(kernel = 'rbf', probability = True, C = 10)
classifier2.fit(X_train, y_train)



# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C' : [1, 5, 10, 50],
               'kernel' : ['rbf', 'linear',]}
             ]
'''{'C' : [50, 75, 100],
               'kernel' : ['poly'], 
               'degree' : [7,9]}'''
'''{'n_estimators' : [501, 502, 600]}'''
'''{'C' : [0.005, 0.01, 0.02, 0.05, 0.1]}  '''

grid_search = GridSearchCV(estimator = classifier2, 
                           param_grid = parameters,
                           scoring = "neg_log_loss",
                           cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_metric = grid_search.best_score_
best_params = grid_search.best_params_



# Predicting the Test Set results
y_pred = classifier2.predict_proba(X_test)

# Writing the results to a csv file
np.savetxt('results.csv', y_pred)

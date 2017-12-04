### HackerEarth - Machine Learning Challenge #2 : Funding Successful Projects on Kickstarter

# Importing the basic libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

## Importing the dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


## EDA - Previewing datasets
train.info()
print("-"*40)
test.info()
''' Findings :
                Train - 1 'name' missing and 8 values in 'desc' column are missing
                Test - 4 missing values in 'desc' column
                'backers_count' column missing from test. Total 12 columns in test
'''                
# Print summary statistics of each column
train.describe() # --> Summary statistics of numerical features + Target
train.describe(include = ['O']) # --> Summary statistics of categorical features
test.describe()
test.describe(include = ['O'])
''' Distribution of 'goal' field similar in train and test, with the test dataset leaning
slightly towards higher values '''


## Create simple features
# Log transformation to make the 'goal' variable more 'Gaussian' like
train['goal'] = np.log10(train['goal'])
test['goal'] = np.log10(test['goal'])

# Creating few text features
train['desc_charLen'] = train['desc'].apply(str).apply(len)
test['desc_charLen'] = test['desc'].apply(str).apply(len)
train['desc_wordCount'] = train['desc'].apply(str).apply(lambda x: len(x.split(' ')))
test['desc_wordCount'] = test['desc'].apply(str).apply(lambda x: len(x.split(' ')))

train['keywords_charLen'] = train['keywords'].apply(str).apply(len)
test['keywords_charLen'] = test['keywords'].apply(str).apply(len)
train['keywords_wordCount'] = train['keywords'].apply(str).apply(lambda x: len(x.split('-')))
test['keywords_wordCount'] = test['keywords'].apply(str).apply(lambda x: len(x.split('-')))

#train['name_wordCount'] = train['name'].apply(str).apply(lambda x: len(x.split(' ')))

# Creating a few time features
train['proj_live_duration(in days)'] = np.round((train['deadline'] - train['launched_at'])/86400.0).astype(int)
test['proj_live_duration(in days)'] = np.round((test['deadline'] - test['launched_at'])/86400.0).astype(int)

train['wait_time_launch(in hrs)'] = np.round((train['launched_at'] - train['created_at'])/3600.0).astype(int)
test['wait_time_launch(in hrs)'] = np.round((test['launched_at'] - test['created_at'])/3600.0).astype(int)

train['diff_deadline_state_change(in mins)'] = np.round((train['state_changed_at'] - train['deadline'])/60.0).astype(int)
test['diff_deadline_state_change(in mins)'] = np.round((test['state_changed_at'] - test['deadline'])/60.0).astype(int)


## Convert time to struct_time structure for better usability
import time
time_cols = ['deadline','state_changed_at','launched_at','created_at']
for x in time_cols:
    train[x] = train[x].apply(lambda k: time.localtime(k))
    test[x] = test[x].apply(lambda k: time.localtime(k))

## Extracting and creating more time-based features 
for t in time_cols: 
    train[t + '_year'] = train[t].apply(lambda k: k.tm_year)
    test[t + '_year'] = test[t].apply(lambda k: k.tm_year)
    
    train[t + '_month'] = train[t].apply(lambda k: k.tm_mon)
    test[t + '_month'] = test[t].apply(lambda k: k.tm_mon)
    
    train[t + '_date'] = train[t].apply(lambda k: k.tm_mday)
    test[t + '_date'] = test[t].apply(lambda k: k.tm_mday)
    
    train[t + '_day'] = train[t].apply(lambda k: k.tm_wday)
    test[t + '_day'] = test[t].apply(lambda k: k.tm_wday)
    
    train[t + '_hour'] = train[t].apply(lambda k: k.tm_hour)
    test[t + '_hour'] = test[t].apply(lambda k: k.tm_hour)

## Creating text features (using CBOW)
import re
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
from nltk.stem.snowball import SnowballStemmer 
from sklearn.feature_extraction.text import CountVectorizer

## creating a full list of descriptions from train and test
desc_corp = pd.Series(train['desc'].tolist() + test['desc'].tolist()).astype(str)
keyword_corp = pd.Series(train['keywords'].tolist() + test['keywords'].tolist()).astype(str)
# Function to clean punctuation, digits, tabs etc.
def text_clean(word):
    p1 = re.sub(pattern='[^a-zA-Z]',repl=' ',string=word)
    p1 = p1.lower()
    return p1

desc_corp = desc_corp.map(text_clean)
keyword_corp = keyword_corp.map(text_clean)

## Splitting, checking for stopwords, Stemming
stop = set(stopwords.words('english'))
desc_corp = [[x for x in x.split() if x not in stop] for x in desc_corp]
keyword_corp = [[x for x in x.split() if x not in stop] for x in keyword_corp]

stemmer = SnowballStemmer(language='english')
desc_corp = [[stemmer.stem(x) for x in x] for x in desc_corp]
keyword_corp = [[stemmer.stem(x) for x in x] for x in keyword_corp]

# Removing words of 2 or lesser characters
desc_corp = [[x for x in x if len(x) > 2] for x in desc_corp]
keyword_corp = [[x for x in x if len(x) > 2] for x in keyword_corp]

desc_corp = [' '.join(x) for x in desc_corp]
keyword_corp = [' '.join(x) for x in keyword_corp]

## Creating features from CBOW model
cv = CountVectorizer(max_features = 500)
desc_features = cv.fit_transform(desc_corp) .todense()
keyword_features = cv.fit_transform(keyword_corp) .todense()

desc_features = pd.DataFrame(desc_features)
desc_features.rename(columns= lambda x: 'desc_'+ str(x), inplace=True)
keyword_features = pd.DataFrame(keyword_features)
keyword_features.rename(columns= lambda x: 'keyword_'+ str(x), inplace=True)

# Split the text features into train and test
desc_train = desc_features[:train.shape[0]]
desc_test = desc_features[train.shape[0]:]
desc_test.reset_index(drop=True,inplace=True)

keyword_train = keyword_features[:train.shape[0]]
keyword_test = keyword_features[train.shape[0]:]
keyword_test.reset_index(drop=True,inplace=True)


## Encoding categorical variables - disable_communication, country, currency
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['disable_communication'] = le.fit_transform(list(train['disable_communication'].values))
test['disable_communication'] = le.transform(list(test['disable_communication'].values))
    
curr_feat = pd.get_dummies(train['currency'].append(test['currency']))
curr_feat.rename(columns = lambda x: 'curr_' + str(x), inplace = True)
curr_feat_train = curr_feat[:train.shape[0]]
curr_feat_test = curr_feat[train.shape[0]:]

country_feat = pd.get_dummies(train['country'].append(test['country']))
country_feat.rename(columns = lambda x: 'country_' + str(x), inplace = True)
country_feat_train = country_feat[:train.shape[0]]
country_feat_test = country_feat[train.shape[0]:]



## Visual EDA
sns.pairplot(train.iloc[:, [6,7]])

train_one = train.loc[(train['final_status'] == 1),:]
train_zero = train.loc[(train['final_status'] == 0),:]

# Comparing histograms of different variables in Success/Failure cases
x = train_one['goal']
y = train_zero['goal']
xweights = 100.0 * np.ones_like(x) / x.size
yweights = 100.0 * np.ones_like(y) / y.size
fig, ax = plt.subplots()
ax.hist(x, bins = 100, range = (0,100), color = 'lightblue', alpha = 0.9, weights = xweights)
ax.hist(y, bins = 100, range = (0,100), color = 'red', alpha = 0.2, weights = yweights)
plt.ylabel('%age of Successful(blue) vs Failed(red) projects')
plt.xlabel('Goal amount')
plt.title('Successful vs Failed projects (by Goal amt.)')





# Preparing X_train, X_test and y_train
features = ['goal', 'country', 'currency', 'proj_live_duration(in days)', 'wait_time_launch(in hrs)']

#features = ['country', 'desc_charLen', 'desc_wordCount', 'keywords_charLen', 'keywords_wordCount',
#            'proj_live_duration(in days)', 'wait_time_launch(in hrs)', 'diff_deadline_state_change(in mins)',
#            'deadline_year', 'deadline_hour', 'launched_at_year', 'launched_at_month',
#            'launched_at_hour', 'created_at_year', 'created_at_month']

X_train = train.loc[:,features]
X_test = test.loc[:,features]
X_train = pd.concat([X_train, country_feat_train, curr_feat_train, desc_train, keyword_train],axis=1).values
X_test = pd.concat([X_test, country_feat_test, curr_feat_test, desc_test, keyword_test],axis=1).values

y_train = train['final_status'].values


### Feature Scaling - used only for SVM classifier; ignored for other classifiers
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)



## Modelling stage
# Model 1 : XGBoost
from xgboost import XGBClassifier
classifier = XGBClassifier(max_depth = 6, learning_rate = 0.01,
                            n_estimators = 300, objective = "binary:logistic", 
                            gamma = 0, base_score = 0.5, reg_lambda = 1, subsample = 0.6,
                            colsample_bytree = 0.8)

classifier.fit(X_train, y_train, eval_metric = "error")
feat_imp = classifier.feature_importances_

# Model 2 : Random Forest classification
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 300, max_depth = 10, 
                                     criterion = "gini", n_jobs = -1)
classifier_rf.fit(X_train, y_train)

# Model 3 : SVM classification
from sklearn.svm import SVC
classifier_svm = SVC(kernel = 'rbf', C = 5, probability = False)
classifier_svm.fit(X_train, y_train)


## Applying GridSearchCV to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C' : [0.1, 0.5, 1, 5, 10, 20]}
             ]

grid_search = GridSearchCV(estimator = classifier_svm, 
                           param_grid = parameters,
                           scoring = None,
                           cv = 10, n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_metric = grid_search.best_score_
best_params = grid_search.best_params_
grid_search.grid_scores_ # See all scores

# Predicting the Test Set results
y_pred = classifier.predict_proba(X_test)[:,1]
y_pred = [1 if x > 0.4 else 0 for x in y_pred]


# Writing the results to a csv file
np.savetxt('results.csv', y_pred)


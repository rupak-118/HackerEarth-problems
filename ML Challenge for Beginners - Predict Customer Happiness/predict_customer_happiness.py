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

# Encode the target variable
train['Is_Response'] = train.Is_Response.map({'happy':1,'not happy':0})


''' Approaches to the problem are listed below '''

### 1. Pre-processing of text features - cleaning, stemming etc.
### 2. Create a word cloud from the text corpus to aid in EDA
### 3. Word embeddings - word2vec, GLoVE, tf-idf, t-SNE
### 4. Feature generation from CBOW and Skip-gram models
### 5. Naive Bayes and Tree-based models
### 6. Simple RNNs/LSTMs
### 7. Ensembles and cross-validation


### EDA - Feature modification, creation, Word Clouds, distributions etc.
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

# Function to clean punctuation, digits, tabs etc.
def text_clean(word):
    p1 = re.sub(pattern='[^a-zA-Z]',repl=' ',string=word)
    p1 = p1.lower()
    return p1

train['Description'] = train.Description.map(text_clean)
test['Description'] = test.Description.map(text_clean)

# Add word count feature
train['wordCount'] = train['Description'].apply(str).apply(lambda x: len(x.split(' ')))
train['wordCount'] = np.log10(train.wordCount)
test['wordCount'] = test['Description'].apply(str).apply(lambda x: len(x.split(' ')))
test['wordCount'] = np.log10(test.wordCount)

sns.distplot(train.wordCount, hist = False)
sns.distplot(test.wordCount, hist = False)

# Shifting target variable to the end
responders = train.Is_Response
train.drop('Is_Response', axis = 1, inplace = True)
train['Is_Response'] = responders

##  Creating a word cloud
from wordcloud import WordCloud, STOPWORDS
# convert all descriptions to a single corpus
corpus = '.'.join(train.Description.tolist())
wordcloud = WordCloud(stopwords = set(STOPWORDS),
                          background_color='white',
                          width=1200,
                          height=1000,
                          max_words = 200,
                          max_font_size = 60
                         ).generate(corpus)

plt.imshow(wordcloud)
plt.axis('off')
plt.show()


## Text-preprocessing from Description variables
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from nltk.stem.snowball import SnowballStemmer

# creating a full list of descriptions from train and test
desc_corp = pd.Series(train['Description'].tolist() + test['Description'].tolist()).astype(str)

# Splitting, removing words of 2 or lesser characters, checking of stopwords, stemming
stop = set(stopwords.words('english'))
desc_corp = [[x for x in x.split() if x not in stop] for x in desc_corp]
desc_corp = [[x for x in x if len(x) > 2] for x in desc_corp]
stemmer = SnowballStemmer(language = 'english')
desc_corp = [[stemmer.stem(x) for x in x] for x in desc_corp]
desc_corp = [' '.join(x) for x in desc_corp]



# Creating feature-arrays
X = train.iloc[:,2:-1].values
y = train.Is_Response.values
X_test = test.iloc[:,2::].values

# Train-validation split
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)

# A very naive CatBoost model
from catboost import CatBoostClassifier
model = CatBoostClassifier(iterations = 500, learning_rate = 0.2, depth = 3, l2_leaf_reg = 20,
                           loss_function = 'Logloss', use_best_model = True)
cat_features = [0,1]
model.fit(X_train, y_train, cat_features, eval_set = (X_val, y_val))

preds = model.predict(X_test)
preds_val = model.predict(X_val)

## Drawing confusion matrix to check model performance
from sklearn.metrics import confusion_matrix
confusion_matrix(y_val, preds_val)
model.score(X_val, y_val)


# XGBoost model
from xgboost import XGBClassifier
classifier = XGBClassifier(max_depth = 2, learning_rate = 0.01,
                            n_estimators = 30, objective = "binary:logistic", 
                            gamma = 0, base_score = 0.5, reg_lambda = 10, subsample = 0.8,
                            colsample_bytree = 0.8)

classifier.fit(X_train, y_train, eval_metric = "error@0.5")
feat_imp = classifier.feature_importances_

preds = model.predict(X_test)
preds_val = model.predict(X_val)
confusion_matrix(y_val, preds_val)






## Submitting predictions
subm = pd.DataFrame({'User_ID': test.User_ID.values  , 'Is_Response': preds})
subm['Is_Response'] = ['happy' if x == 1 else 'not happy' for x in subm['Is_Response']]
subm = subm[['User_ID','Is_Response']]
subm.to_csv('sub03.csv', index = False)













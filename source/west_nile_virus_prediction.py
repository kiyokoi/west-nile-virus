### Capstone Project
### Machine Learning Engineer Nanodegree
### West Nile Virus Prediction main code
### Kiyoko Ikeuchi   Auguest, 2016

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import callables as cl
from IPython.display import display

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import cross_val_score, ShuffleSplit, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc

pd.set_option('display.max_columns', 50)

# Load data
train = pd.read_csv('../input/train.csv')
weather = pd.read_csv('../input/weather.csv', na_values=['M', '-', ' '])

# Get to know what's in the dataset
display(train.head(), train.shape)
display(weather.columns.values, weather.shape)

### Data Exploration
# Check the statistic of the training data
display(train.describe())
display(train.Species.unique())

# Exclude redundant columns from the training data
train_drop = ['Address', 'AddressNumberAndStreet', 'AddressAccuracy']
train = train.drop(train_drop, axis=1)

# Check data types
display(train.dtypes)
display(weather.dtypes)

### Weather data missing data
# Count the number of missing data by column
for column in weather.columns:
    print column, weather[column].isnull().values.sum()

# Count the number of missing data by station
print 1472 - weather.groupby('Station').count()

# Drop columns with high count of missing data in both stations and unimputable Station2 data
weather_drop = ['Water1', 'CodeSum']
weather = weather.drop(weather_drop, axis=1)

# Impute 'T' (trace) precipitation with zero
weather.replace('  T', 0.0, inplace=True)
# Convert PrecipTotal to real number
weather['PrecipTotal'] = weather['PrecipTotal'].astype(float)

# Impute missing data with median for: Tavg, WetBulb, Heat, Cool, PrecipTotal, StnPressure, SeaLevel, AvgSpeed
weather_impute = ['Tavg', 'WetBulb', 'Heat', 'Cool', 'PrecipTotal', 'StnPressure', 'SeaLevel', 'AvgSpeed']
# Data statistics before imputation
display(weather[weather_impute].describe())
for col in weather_impute:
    median = weather[col].median()
    weather[col].loc[weather[col].isnull()] = median
    
# Data statistics after imputation
display(weather[weather_impute].describe())

# Verify all missing data has been accounted for
for column in weather.columns:
    print column, weather[column].isnull().values.sum()

# First 5 rows of the final weather data
display(weather.head())

# Convert Date to datetime format and add Year & Month columns
files = [train, weather]
for file in files:
    file['Date'] = pd.to_datetime(file['Date'], format='%Y-%m-%d')
    file['Year'] = file['Date'].dt.year
    file['Month'] = file['Date'].dt.month
    file['Day'] = file['Date'].dt.day
    
### Visualizations
# Visualize locations on the map
cl.Map(train)

# Visualize weather data
var = ['Tavg', 'DewPoint', 'PrecipTotal', 'StnPressure', 'Month', 'Station']
sns.pairplot(weather[var], hue='Station')
plt.savefig('../working/pairplot.png')

# Visualizing virus presence trend
fig = plt.figure(figsize=(12,8))
groups = ['Year', 'Month', 'Species', 'Trap', 'Block']
cl.PercPos(fig, train, groups)
fig.tight_layout()
fig.show()
plt.savefig('../working/histograms.png')

### Data Processing
# Combined weather data by Station, then to train by Date
weather_stn1 = weather[weather['Station']==1]
weather_stn2 = weather[weather['Station']==2]
weather_stn1 = weather_stn1.drop('Station', axis=1)
weather_stn2 = weather_stn2.drop('Station', axis=1)
weather = weather_stn1.merge(weather_stn2, on='Date')

train = train.merge(weather, on='Date')
train = train.drop(['Date', 'Sunrise_y', 'Sunset_y', 'Depart_y', 'SnowFall_y', 'Depth_y', 'Year_y', 'Month_y', 'Day_y'], axis=1)
display(train.head())

# Process categorical features
cat = ['Species', 'Street', 'Trap']
lbl = LabelEncoder()
for col in cat:
    lbl.fit(list(train[col].values))
    train[col] = lbl.transform(train[col].values)

# Identify features and labels
target_col = 'WnvPresent'
x_all = train.drop([target_col], axis=1)
y_all = train[target_col]
features = x_all.values
labels = y_all.values

### Model Application
names = ['GaussianNB', 'Decision Tree', 'Random Forest']
alg = [GaussianNB(), DecisionTreeClassifier(), RandomForestClassifier()]

cv = ShuffleSplit(features.shape[0], n_iter = 10, test_size = 0.2, random_state = 0)

clf_dict = {}
for i in range(len(names)):
    clf_dict[names[i]] = alg[i]

for name, clf in clf_dict.iteritems():   
    score = cross_val_score(clf, features, labels, cv=cv, scoring='roc_auc')
    print '{} score: {:.2f}'.format(name, score.mean())

### Analyzing model performance
# Learning Curve for Gaussian Naive Bayes
clf = GaussianNB()
cl.LearningCurve(clf, features, labels)
plt.suptitle('Learning Curve for Gaussian Naive Bayes', size=14)
plt.savefig('../working/learning_curve_nb.png')

# Learning curve for Decision Tree Classifier
fig = plt.figure(figsize=(10,7))
max_depth = [2, 4, 6, 8]
for k, depth in enumerate(max_depth):
    clf = DecisionTreeClassifier(max_depth = depth)
    fig.add_subplot(2, 2, k+1)
    plt.title('max_depth = %s'%(depth))
    cl.LearningCurve(clf, features, labels)
fig.tight_layout()
fig.subplots_adjust(top=0.90)
fig.show()
plt.suptitle('Learning Curve for Decision Tree Classifier', size=14)
plt.savefig('../working/learning_curve_dt.png')

# Learning curve for Random Forest Classifier
fig = plt.figure(figsize=(10,7))
max_depth = [2, 4, 6, 8]
for k, depth in enumerate(max_depth):
    clf = RandomForestClassifier(max_depth = depth)
    fig.add_subplot(2, 2, k+1)
    plt.title('max_depth = %s'%(depth))
    cl.LearningCurve(clf, features, labels)
fig.tight_layout()
fig.subplots_adjust(top=0.90)
fig.show()
plt.suptitle('Learning Curve for Random Forest Classifier', size=14)
plt.savefig('../working/learning_curve_rf.png')

# Model refinement with Gridsearch
cv = ShuffleSplit(features.shape[0], n_iter = 10, test_size = 0.3, random_state = 0)
clf = RandomForestClassifier()
select = SelectKBest()
steps = [('feature_selection', select), ('rf', clf)]
parameters = dict(feature_selection__k=[5,15,25,35,'all'], rf__max_depth=[4,6,8], rf__min_samples_split=[2, 4, 6, 8])
pipeline = Pipeline(steps)
grid_search = GridSearchCV(pipeline, param_grid=parameters, cv=cv, scoring='roc_auc')
grid_search.fit(features, labels)
print 'Best score: {}'.format(grid_search.best_score_)
print 'best parameters: {}'.format(grid_search.best_params_)

# The final model
# Select 25 best features
display(features.shape)
select = SelectKBest(k=25)
features = select.fit_transform(features, labels)
display(features.shape)

# Model with 25 features
clf = RandomForestClassifier(max_depth=6, min_samples_split=6)
score = cross_val_score(clf, features, labels, cv=cv, scoring='roc_auc')
print 'Avg Score: {:.2f}'.format(score.mean())

# Verify results. Check for overfitting (score dependency on the training/testing subsets) and repeatability.
# check for overfitting (score dependency on the training/testing subsets)
model_accuracies = []
for repetition in range(100):
    (features_train, features_test, labels_train, labels_test) = train_test_split(features, labels, train_size=0.7)
    clf.fit(features_train, labels_train)
    pred = clf.predict_proba(features_test)[:,1]
    score = roc_auc_score(labels_test, pred)
    model_accuracies.append(score)

sns.distplot(model_accuracies)
plt.title('Histogram of Performance over 100 runs', size=14)
plt.xlabel('Score')
plt.ylabel('Count')
plt.savefig('../working/model_accuracy.png')

# Visualize ROC curve
clf.fit(features_train, labels_train)
pred = clf.predict_proba(features_test)[:,1]

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(labels_test, pred)
    roc_auc[i] = auc(fpr[i], tpr[i])
    
plt.figure()
plt.plot(fpr[1], tpr[1])
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve for cross-validation data', size=14)
plt.savefig('../working/roc_curve.png')

# Compare the model against Kaggle benchmark
data = pd.read_csv('../input/predict-west-nile-virus_public_leaderboard.csv')

teams = data.groupby(['TeamName'])
max_scores = teams['Score'].max()

display(max_scores.describe())
sns.distplot(max_scores, label='Leaderboard Scores')
sns.distplot(model_accuracies, label='Scores from 100 runs')
plt.title('Model performance against Kaggle Leaderboard', size=14)
plt.legend()
plt.savefig('../working/leaderboard.png')

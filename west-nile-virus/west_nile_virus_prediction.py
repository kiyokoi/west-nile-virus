# Code is borrowed from this script: https://www.kaggle.com/users/213536/vasco/predict-west-nile-virus/west-nile-heatmap

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Load data
train = pd.read_csv('../input/train.csv')
weather = pd.read_csv('../input/weather.csv', na_values=['M', '-', ' '])

# Get to know what's in the dataset
display(train.head(), train.shape)
display(weather.columns.values, weather.shape)

# Data Exploration
# Exclude redundant columns from the training data
train_drop = ['Address', 'AddressNumberAndStreet', 'AddressAccuracy']
train = train.drop(train_drop, axis=1)

# Check the statistic of the training data
display(train.describe())
display(train.Species.unique())

# Check data types
display(train.dtypes)
display(spray.dtypes)
display(weather.dtypes)

# Tidy up the weather data
# Count the number of missing data by column
for column in weather.columns:
    print column, weather[column].isnull().values.sum()

# Count the number of missing data by station
print 1472 - weather.groupby('Station').count()

# Drop columns with high count of missing data in both stations and unimputable Station2 data
weather_drop = ['Water1', 'CodeSum', 'Depart', 'SnowFall', 'Depth']
weather = weather.drop(weather_drop, axis=1)

# Impute Station2 data with Station1 data for sunrise & sunset
weather_ffill = ['Sunrise', 'Sunset']
for column in weather_ffill:
    weather[column].fillna(method='ffill', inplace=True)

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
    weather.loc[weather[col].isnull()] = median
    
# Data statistics after imputation
display(weather[weather_impute].describe())

# Verify all missing data has been accounted for
for column in weather.columns:
    print column, weather[column].isnull().values.sum()

# First 5 rows of the final weather data
display(weather.head())

# Visualize locations on the map
# Modified from Kaggle starter code
mapdata = np.loadtxt("../input/mapdata_copyright_openstreetmap_contributors.txt")
traps = pd.read_csv('../input/train.csv')[['Date', 'Trap','Longitude', 'Latitude', 'WnvPresent']]

aspect = mapdata.shape[0] * 1.0 / mapdata.shape[1]
lon_lat_box = (-88, -87.5, 41.6, 42.1)

plt.figure(figsize=(10,14))
plt.imshow(mapdata, 
           cmap=plt.get_cmap('gray'), 
           extent=lon_lat_box, 
           aspect=aspect)

locations = traps[['Longitude', 'Latitude']].drop_duplicates().values
stations = np.array([[-87.933, 41.995], [-87.752, 41.786, ]])
plt.scatter(locations[:,0], locations[:,1], marker='o')
for i in range(2):
    plt.plot(stations[i,0], stations[i,1], '^', color='r')

# Convert Date to datetime format and add Year & Month columns
files = [train, weather]
for file in files:
    file['Date'] = pd.to_datetime(file['Date'], format='%Y-%m-%d')
    file['Year'] = file['Date'].dt.year
    file['Month'] = file['Date'].dt.month
    file['Day'] = file['Date'].dt.day
    
# Determine percentage of positive mosquitos per category
groups = ['Year', 'Month', 'Species', 'Trap', 'Block']
for i in range(len(groups)):
    group = train.groupby([groups[i]])
    percent_positive_per_group = dict(group['WnvPresent'].sum() / group['WnvPresent'].count() * 100.0)
    plt.bar(range(len(percent_positive_per_group)), percent_positive_per_group.values(), align = 'center')
    plt.xticks(range(len(percent_positive_per_group)), percent_positive_per_group.keys())
    locs, labels = plt.xticks()
    plt.setp(labels, rotation = 70)
    plt.show()

### Data Processing
# Combined weather data by Station, then to train by Date
weather_stn1 = weather[weather['Station']==1]
weather_stn2 = weather[weather['Station']==2]
weather_stn1 = weather_stn1.drop('Station', axis=1)
weather_stn2 = weather_stn2.drop('Station', axis=1)
weather = weather_stn1.merge(weather_stn2, on='Date')

train = train.merge(weather, on='Date')
train = train.drop(['Date'], axis=1)
display(train.head())

# This is an example of developing a script locally with the West Nile Virus data to share on Kaggle
# Once you have a script you're ready to share, paste your code into a new script at:
#	https://www.kaggle.com/c/predict-west-nile-virus/scripts/new

# Code is borrowed from this script: https://www.kaggle.com/users/213536/vasco/predict-west-nile-virus/west-nile-heatmap

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# Load data
train = pd.read_csv('../input/train.csv')
spray = pd.read_csv('../input/spray.csv')
weather = pd.read_csv('../input/weather.csv', na_values=['M', '-', ' '])

# Get to know what's in the dataset
display(train.head(), train.shape)
display(spray.head(), spray.shape)
display(weather.columns.values, weather.shape)

# Initial visualization
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

#drop rows with off-the-map locations
spray = spray[spray.Latitude < 42.2]
spray_locations = spray[['Longitude', 'Latitude']].drop_duplicates().values
locations = traps[['Longitude', 'Latitude']].drop_duplicates().values
plt.scatter(spray_locations[:,0], spray_locations[:,1], marker='o', color = 'c')
plt.scatter(locations[:,0], locations[:,1], marker='x')

# Data Exploration
# Exclude redundant columns from the training data
train_drop = ['Address', 'Block', 'Street', 'AddressNumberAndStreet', 'AddressAccuracy']
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
# Data statistics after imputation
display(weather[weather_impute].describe())

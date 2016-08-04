# Predict West Nile virus in mosquitos across the city of Chicago
This is a capstone project for Udacity's Machine Learning Engineer Nanodegree. Project scope and input data are from [Kaggle's completed competition archive](https://www.kaggle.com/c/predict-west-nile-virus).

### Code
Code is provided in `west_nile_virus_prediction.py` and `west_nile_virus_prediction.ipynb`. Auxilary codes can be found in `callables.py`.

This program requres Python 2.7 and the following Python libraries installed:
* NumPy
* Pandas
* matplotlib
* Seaborn
* Scikit-learn

### Data
Dataset used in thie project is included as `train.csv` and `weather.csv`. Dataset was provided by Kaggle and contains the following attributes.

train.csv - training dataset consisting of data from 2007, 2009, 2011, and 2013:
* `Id`: the id of the record
* `Date`: date that the WNV test is performed
* `Address`: approximate address of the location of trap. This is used to send to the GeoCoder. 
* `Species`: the species of mosquitos
* `Block`: block number of address
* `Street`: street name
* `Trap`: Id of the trap
* `AddressNumberAndStreet`: approximate address returned from GeoCoder
* `Latitude`, `Longitude`: Latitude and Longitude returned from GeoCoder
* `AddressAccuracy`: accuracy returned from GeoCoder
* `NumMosquitos`: number of mosquitoes caught in this trap
* `WnvPresent`: whether West Nile Virus was present in these mosquitos. 1 means WNV is present, and 0 means not present. 

weather.csv - weather data from 2007 to 2014:
* `Station`: station ID
* `Date`: date of weather recorded
* `Tmax`, `Tmin`, `Tave`: temperature in Fahrenheit
* `Depart`: departure from normal
* `DewPoint`: average dew point
* `WetBulb`: average wet bulb
* `Heat`, `Cool`: index
* `Sunrise`, `Sunset`: calculated, not observed
* `PrecipTotal`: precipitation in inches
* `SnowFall`: snow on ground in inches
* `StnPressure`: average station pressure
* `Sealevel`: average seal level pressure
* `ResultSpeed`, `ResultDir`, `AvgSpeed`: wind speed in mph, direction in degrees

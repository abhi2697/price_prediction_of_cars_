# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 17:03:26 2023

@author: abppa
"""

#PRIDICTING PRICE OF PRE-OWNED CARS
import pandas as pd
import numpy as np
import seaborn as sns

#setting dimensions for plot
sns.set(rc={'figure.figsize':(11.7,8.27)})
#reading CSV file
df1=pd.read_csv('cars_sampled.csv') 
#creating copy
df2=df1.copy()
#structure of the dataset
df2.info()

#summarizing data
df2.describe()
pd.set_option('display.float_format',lambda x: '%.3f' %x)#only in 3 values
df2.describe()
# to display maximum set of columns
pd.set_option('display.max_columns', 500)
df2.describe()
#dropping unwanted columns
col=['name','dateCrawled','dateCreated','postalCode','lastSeen']
df2=df2.drop(columns=col,axis=1)
#removing duplicate records
df2.drop_duplicates(keep='first',inplace=True)
#470 duplicates
#data cleaning
#No. of missing values in each column
df2.isnull().sum()
#Variables yearOfRegistration
yearwise_count=df2['yearOfRegistration'].value_counts().sort_index()
#yearwise_count iska matlab ye hai ik saal kitni gaadiyan hai

sum(df2['yearOfRegistration']>2018)
sum(df2['yearOfRegistration']<1950)
sns.regplot(x='yearOfRegistration',y='price',scatter=True,fit_reg=False,data=df2)
#working range -1950 and 2018
price_count=df2['price'].value_counts().sort_index()
sns.distplot(df2['price'])
df2['price'].describe()
sns.boxplot(y=df2['price'])
sum(df2['price']>150000)
sum(df2['price']<100)
#working range -100 and 150000

#Variable powerPS
power_count=df2['powerPS'].value_counts().sort_index()
sns.distplot(df2['powerPS'])
df2['powerPS'].describe()
sns.boxplot(y=df2['powerPS'])
sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=False,data=df2)
sum(df2['powerPS']>500)
sum(df2['powerPS']< 10)
#working range -10 and 500
#setting working range of data

df2=df2[(df2.yearOfRegistration <=2018)&
        (df2.yearOfRegistration >=1950)
        &(df2.price >=100)
        &(df2.price <=150000)
        &(df2.powerPS >=10)
        &(df2.powerPS <=500)]
# -6700 records are dropped
df2['monthOfRegistration']/=12
df2['monthOfRegistration']=df2['monthOfRegistration'].round(decimals=2)

#Creating new variable age by adding yearOfRegistration and monthOfRegistration
df2['Age']=(2018 - df2['yearOfRegistration'])+df2['monthOfRegistration']
df2['Age'].describe()

#dropping YearOfRegistration and monthOfRegistration
df2=df2.drop(columns=['yearOfRegistration','monthOfRegistration'],axis=1)

#Visualizing parameters

#Age
sns.distplot(df2['Age'])
sns.boxplot(y=df2['Age'])

#price
sns.distplot(df2['price'])
sns.boxplot(y=df2['price'])

#powerPS
sns.distplot(df2['powerPS'])
sns.boxplot(y=df2['powerPS'])

#visualizing parameters after narrowing working
#age vs price
sns.regplot(x='Age',y='price',scatter=True,fit_reg=False,data=df2)
#Cars priced higher are newer
#with increase in age,price decrease
#however some cars are priced higher with increased in age

#powerPS vs price
sns.regplot(x='powerPS',y='price',scatter=True,fit_reg=False,data=df2)

#Variable seller 
df2['seller'].value_counts()
pd.crosstab(df2['seller'], columns='count',normalize=True)
sns.countplot(x='seller', data=df2)
#fewer cars have 'commercial' => insignificant

#variable offerType
df2['offerType'].value_counts()
sns.countplot(x='offerType',data=df2)
#all cars have offer =>insignificant

#Variable abtest
df2['abtest'].value_counts()
pd.crosstab(df2['abtest'], columns='count',normalize=True)
sns.countplot(x='abtest',data=df2)

#Equally distributed
sns.boxplot(x='abtest',y='price',data=df2)
#for every price value there is almost 50-50 distributed
#does not affect price=> Insignificant
#Variable vehicle type

df2['vehicleType'].value_counts()
pd.crosstab(df2['vehicleType'], columns='count',normalize=True)
sns.countplot(x='vehicleType',data=df2)
sns.boxplot(x='vehicleType',y='price', data=df2)
# vehicle type does effect price

#variable gearbox
df1['gearbox'].value_counts()
pd.crosstab(df2['gearbox'], columns='count',normalize=True)
sns.countplot('gearbox',data=df2)
sns.boxplot(x='gearbox',y='price',data=df2)
#gearbox affects price

#variable model
df1['model'].value_counts()
pd.crosstab(df2['model'],columns='count',normalize=True)
sns.countplot(x='model',data=df2)
sns.boxplot(x='kilometer',y='price',data=df2)
#considered in modelling
#variable kilometer
df2['kilometer'].value_counts().sort_index()
pd.crosstab(df2['kilometer'], columns='count',normalize=True)
sns.boxplot(x='kilometer', y='price',data=df2)
df2['kilometer'].describe()
sns.distplot(df2['kilometer'],bins=8,kde=False)
sns.regplot(x='kilometer',y='price',scatter=True,fit_reg=False,data=df2)
#considered in modelling

#variable fuelType
df2['fuelType'].value_counts()
pd.crosstab(df2['fuelType'], columns='count',normalize=True)
sns.countplot(x='fuelType', data=df2)
sns.boxplot(x='fuelType', y='price',data=df2)
#fuelType affects price

#variable brand 
df2['brand'].value_counts()
pd.crosstab(df2['brand'], columns='count',normalize=True)
sns.countplot(x='brand',data=df2)
sns.boxplot(x='brand',y='price',data=df2)

#Cars are distributed over many brands
#considered for modelling

#Variable  notrepaired
#yes -car is damaged but not rectified
#no-car was damaged but has been rectified
df2['notRepairedDamage'].value_counts()
pd.crosstab(df2['notRepairedDamage'], columns='count',normalize=True)
sns.countplot(x='notRepairedDamage', data=df2)
sns.boxplot(x='notRepairedDamage', y='price',data=df2)
#as expected,that cars that require the damages to be repaired
#fall under lower price ranges
col=['seller','offerType','abtest']
df2=df2.drop(columns=col,axis =1)
df3=df2.copy()

df_select=df2.select_dtypes(exclude=[object])
correlation=df_select.corr()
round(correlation,3)
df_select.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]


"""
we are going to build a linear regression and random forest model
on two sets of data.
1.data obtained by omitting rows with any missing value
2.data obtained by imputting the missing value
"""
df_omit=df2.dropna(axis=0)

#Converting categorical variables
df_omit=pd.get_dummies(df_omit,drop_first=True)
#importing necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#
#model building with omitted data
#
 
#separting input and output features
x1=df_omit.drop(['price'],axis='columns',inplace=False)
y1=df_omit['price']

#plotting the variable price
prices=pd.DataFrame({"1.Before":y1,"2.After":np.log(y1)})
prices.hist()

#transforming price as a logarithmic value
y1=np.log(y1)

#splitting data into test and train
X_train,X_test,y_train,y_test=train_test_split(x1,y1,test_size=0.3,random_state=3)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

#Baseline model for omitted data

#finding the mean for test data value
base_pred=np.mean(y_test)
print(base_pred)

#repeating some value till length of test data
base_pred=np.repeat(base_pred, len(y_test))

#finding the RMSE
base_root_mean_square_error=np.sqrt(mean_squared_error(y_test, base_pred))
print(base_root_mean_square_error)

#linear regression with omitted data

#setting intercept as true
lgr=LinearRegression(fit_intercept=True)

#model
model_lr1=lgr.fit(X_train,y_train)
#predicting model on test set
df_prediction_ln1=lgr.predict(X_test)

#computing MSE and Rmse
lin_mse1=mean_squared_error(y_test,df_prediction_ln1)
lin_rmse1=np.sqrt(lin_mse1)
print(lin_rmse1)

#R squared value
r2_Lin_test1=model_lr1.score(X_test,y_test)
r2_lin_train1=model_lr1.score(X_train,y_train)
print(r2_Lin_test1,r2_lin_train1)

#regression diagnostics :- residual plot analysis
residuals1=y_test-df_prediction_ln1
sns.regplot(x=df_prediction_ln1,y=residuals1,scatter=True,fit_reg=True)
residuals1.describe()

#random forest with omitted data
#Model parameters
rf=RandomForestRegressor(n_estimators=100,max_depth=100,max_features='auto',min_samples_split=10,min_samples_leaf=4,random_state=1)
model_rf1=rf.fit(X_train,y_train)

#predicting model on train set
df_prediction_rf1=rf.predict(X_test)
#computing MSE and RMSE
rf_mse1=mean_squared_error(y_test, df_prediction_rf1)
rf_rmse1=np.sqrt(rf_mse1)
print(rf_rmse1)

#R squared value
r2_rf_test1=model_rf1.score(X_test,y_test)
r2_rf_train1=model_rf1.score(X_train,y_train)
print(r2_rf_test1,r2_rf_train1)

#model buildin with imputed data
df_imputed=df2.apply(lambda x:x.fillna(x.median()) if x.dtype=='float' else x.fillna(x.value_counts().index[0]))
df_imputed.isnull().sum()
#converting categorial variables to dummy variables
df_imputed=pd.get_dummies(df_imputed, drop_first=True)
#separting input and output features
x2=df_imputed.drop(['price'],axis='columns',inplace=False)
y2=df_imputed['price']

#plotting the variable price
prices=pd.DataFrame({"1.Before":y1,"2.After":np.log(y2)})
prices.hist()

#transforming price as a logarithmic value
y2=np.log(y2)

#splitting data into test and train
X_train1,X_test1,y_train1,y_test1=train_test_split(x2,y2,test_size=0.3,random_state=3)
print(X_train1.shape,X_test1.shape,y_train1.shape,y_test1.shape)


#Baseline model for omitted data

#finding the mean for test data value
base_pred1=np.mean(y_test1)
print(base_pred1)

#repeating some value till length of test data
base_pred1=np.repeat(base_pred1, len(y_test1))

#finding the RMSE
base_root_mean_square_error2=np.sqrt(mean_squared_error(y_test1, base_pred1))
print(base_root_mean_square_error2)

#linear regression with omitted data

#setting intercept as true
lgr2=LinearRegression(fit_intercept=True)

#model
model_lr2=lgr2.fit(X_train1,y_train1)
#predicting model on test set
df_prediction_ln2=lgr2.predict(X_test1)

#computing MSE and Rmse
lin_mse2=mean_squared_error(y_test1,df_prediction_ln2)
lin_rmse2=np.sqrt(lin_mse1)
print(lin_rmse2)

#R squared value
r2_Lin_test2=model_lr2.score(X_test1,y_test1)
r2_lin_train2=model_lr2.score(X_train1,y_train1)
print(r2_Lin_test2,r2_lin_train2)

#regression diagnostics :- residual plot analysis
residuals2=y_test1-df_prediction_ln2
sns.regplot(x=df_prediction_ln2,y=residuals2,scatter=True,fit_reg=True)
residuals2.describe()

#random forest with omitted data
#Model parameters
rf2=RandomForestRegressor(n_estimators=100,max_depth=100,max_features='auto',min_samples_split=10,min_samples_leaf=4,random_state=1)
model_rf2=rf2.fit(X_train1,y_train1)

#predicting model on train set
df_prediction_rf2=rf2.predict(X_test1)
#computing MSE and RMSE
rf_mse2=mean_squared_error(y_test1, df_prediction_rf2)
rf_rmse2=np.sqrt(rf_mse2)
print(rf_rmse1)

#R squared value
r2_rf_test2=model_rf2.score(X_test1,y_test1)
r2_rf_train2=model_rf2.score(X_train1,y_train1)
print(r2_rf_test2,r2_rf_train2)


print("Metrics for models built from data where missing values were omitted")
print("R squared value for train from linear Regression = %s "% r2_lin_train1)
print("R squared value for test from linear Regression = %s "% r2_Lin_test1)
print("R squared value for train from random forest = %s "% r2_rf_train1)
print("R squared value for test from random forest = %s "% r2_rf_test1)
print("Base RMSE of model built from data where missing values were omitted= %s"%base_root_mean_square_error)
print("RMSE value for test from Linear Regression= %s" %lin_rmse1)   
print("RMSE value for test from Linear Regression= %s" %rf_rmse1) 
print("\n \n")  
print("Metrics for models built from data where missing values were not  omitted")
print("R squared value for train from linear Regression = %s "% r2_lin_train2)
print("R squared value for test from linear Regression = %s "% r2_Lin_test2)
print("R squared value for train from random forest = %s "% r2_rf_train2)
print("R squared value for test from random forest = %s "% r2_rf_test2)
print("Base RMSE of model built from data where missing values were omitted= %s"%base_root_mean_square_error2)
print("RMSE value for test from Linear Regression= %s" %lin_rmse2)   
print("RMSE value for test from Linear Regression= %s" %rf_rmse2) 
print("\n \n")  

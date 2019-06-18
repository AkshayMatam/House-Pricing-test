# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:51:12 2019

@author: aksham
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

# Put this when it's called
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


#This is to find how the missing data is distributed 
def draw_missing_data_table(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)*100
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data

# Plot learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Validation score")

    plt.legend(loc="best")
    return plt

# Plot validation curve
def plot_validation_curve(estimator, title, X, y, param_name, param_range, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    train_scores, test_scores = validation_curve(estimator, X, y, param_name, param_range, cv)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(param_range, train_mean, color='r', marker='o', markersize=5, label='Training score')
    plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='r')
    plt.plot(param_range, test_mean, color='g', linestyle='--', marker='s', markersize=5, label='Validation score')
    plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='g')
    plt.grid() 
    plt.xscale('log')
    plt.legend(loc='best') 
    plt.xlabel('Parameter') 
    plt.ylabel('Score') 
    plt.ylim(ylim)

def getChi2(feat1,feat2):
    import scipy
    observed=pd.crosstab(feat1,feat2)
    output=scipy.stats.chi2_contingency(observed)
    chi2_stat=output[0]
    p_value=output[1]
    dof=output[2]
    expected_values=output[3]
    return observed,chi2_stat,p_value,dof,expected_values
    
data=pd.read_csv('train.csv',encoding='iso--8859-1', low_memory = False)

cols=data.columns
num_cols = data._get_numeric_data().columns
cat_cols = list(set(cols) - set(num_cols))


#Insights on Numerical Data
data_description=data.describe()


#histogram
sns.distplot(data['SalePrice'])

#skewness and kurtosis
print("Skewness: %f" % data['SalePrice'].skew())
print("Kurtosis: %f" % data['SalePrice'].kurt())


corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

#Top 20 correlated features for SalePrice

corr_cols=corrmat.nlargest(20,'SalePrice')['SalePrice'].index
cm = np.corrcoef(data[corr_cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=corr_cols.values, xticklabels=corr_cols.values)
plt.show()


data_train=data[corr_cols]
#After looking at the heat map we can delete correlated columns
#TotalBsmtSF,GarageCars,TotRmsAbvGrid,GarageYrBlt

data_train.drop(['TotalBsmtSF','GarageArea','TotRmsAbvGrd'],axis=1,inplace=True) 

missing_data_analysis=draw_missing_data_table(data)


#drop the  feature if the missingdata is more than 75%
data.drop(['PoolQC','MiscFeature','Alley','Fence','Id'],axis=1,inplace=True)


cols=data_train.columns
num_cols = data_train._get_numeric_data().columns
cat_cols = list(set(cols) - set(num_cols))

for col in data_train.columns:
    if (data[col].skew()>0.8 or data[col].skew()<-0.8):
        print(col)
        data[col]=data[col].fillna(0)
        data['HasCol'] = pd.Series(len(data[col]), index=data.index)
        data['HasCol']=0
        data.loc[data[col]>0,'HasCol']=1
        data.loc[data['HasCol']==1,col]=np.log(data[col])
        data_train[col]=data[col]




data_train=pd.concat([data_train,data[cat_cols]],axis=1)

#data_train=pd.concat([data['SalePrice'],data['OverallQual'],data['GrLivArea'],data['GarageCars'],data['GarageArea'],
#                      data['TotalBsmtSF'],data['FullBath'],data['TotRmsAbvGrd'],data['YearBuilt'],data['YearRemodAdd'],
#                      data['Fireplaces'],data['BsmtFinSF1'],data['WoodDeckSF'],data['2ndFlrSF'],data['OpenPorchSF'],data['HalfBath'],
#                      data['LotShape'],data['LotConfig'],data['Neighborhood'],data['HouseStyle']],axis=1)

 




sns.distplot((data_train['YearBuilt']))






#scatter plot between variables and salesprice to check the relation
var='LotFrontage'
data1 = pd.concat([data['SalePrice'], data[var]], axis=1)
data1.plot.scatter(x=var,y='SalePrice')

#Transforming the data using log
#Here in 2ndflrSF column we have 0's so log transformation will not have good impact
# So , Transforming only the data above 0
data['Has2ndFlr'] = pd.Series(len(data['2ndFlrSF']), index=data.index)
data['Has2ndFlr']=0
data.loc[data['2ndFlrSF']>0,'Has2ndFlr']=1
data.loc[data['Has2ndFlr']==1,'2ndFlrSF']=np.log(data['2ndFlrSF'])

data['LotFrontage']=data['LotFrontage'].fillna(0)
data['HasLotFrontage'] = pd.Series(len(data['LotFrontage']), index=data.index)
data['HasLotFrontage']=0
data.loc[data['LotFrontage']>0,'HasLotFrontage']=1
data.loc[data['HasLotFrontage']==1,'LotFrontage']=np.log(data['LotFrontage'])

data_train['SalePrice']=np.log(data['SalePrice'])
data_train['LotFrontage']=data['LotFrontage']
data_train['2ndFlrSF']=data['2ndFlrSF']






##### Categorical Data EDA ####

var='LotShape'
data1 = pd.concat([data['SalePrice'], data[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.swarmplot(x=var, y="SalePrice", data=data1)
#fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);

var='LotConfig'
data1 = pd.concat([data['SalePrice'], data[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.swarmplot(x=var, y="SalePrice", data=data1)
#fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);

var='Neighborhood'
data1 = pd.concat([data['SalePrice'], data[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.swarmplot(x=var, y="SalePrice", data=data1)
#fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);


var='HouseStyle'
data1 = pd.concat([data['SalePrice'], data[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.swarmplot(x=var, y="SalePrice", data=data1)
#fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);

var='OverallQual'
data1 = pd.concat([data['SalePrice'], data[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.swarmplot(x=var, y="SalePrice", data=data1)
#fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);


var='YearBuilt'
data1 = pd.concat([data['SalePrice'], data[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.swarmplot(x=var, y="SalePrice", data=data1)
#fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);

var='LandContour'
data1 = pd.concat([data['SalePrice'], data[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.swarmplot(x=var, y="SalePrice", data=data1)
#fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);

var='OverallQual'
data1 = pd.concat([data['SalePrice'], data[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.swarmplot(x=var, y="SalePrice", data=data1)
#fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);



data_train.drop(['LandContour','MSZoning','Street','Utilities','LandSlope'],axis=1,inplace=True) 

for col in cat_cols:
    data_train[col].fillna(data_train[col].mode()[0],inplace=True)

data_train.dropna(inplace=True)

Features= data_train.drop(['SalePrice'],axis=1)
Target= data_train['SalePrice']

Features=pd.get_dummies(Features,drop_first=True)



X_train,X_test,y_train,y_test=train_test_split(data['OverallQual'],data['SalePrice'],test_size=0.2,random_state=0)



from sklearn.linear_model import LinearRegression

regressor= LinearRegression()
regressor.fit(X_train.reshape(-1,1),y_train)



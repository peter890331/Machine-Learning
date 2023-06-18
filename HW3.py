"""
Machine Learning HW3: Programming

@author: Peter
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')         # pycharm bug
import matplotlib.pyplot as plt
#import pymc3 as pm             # useless, doesn't work ==
from sklearn.linear_model import LinearRegression

df_exercise = pd.read_csv('exercise.csv')                                                                                           # load exercise.csv
for i in range(0,len(df_exercise)):                                                                                                 # change Gender description to binary
    if df_exercise.loc[i,'Gender'] == 'male':
        df_exercise.loc[i, 'Gender'] = 1
    else:
        df_exercise.loc[i, 'Gender'] = 0
#normalized_df_exercise = pd.concat([df_exercise.iloc[:,0:2],(df_exercise.iloc[:,2:]-df_exercise.iloc[:,2:].mean())/df_exercise.iloc[:,2:].std()],axis=1)
normalized_df_exercise = pd.concat([df_exercise.iloc[:,0:2],(df_exercise.iloc[:,2:]-df_exercise.iloc[:,2:].min())/(df_exercise.iloc[:,2:].max()-df_exercise.iloc[:,2:].min())],axis=1)          # exercise normalization

df_calories = pd.read_csv('calories.csv')                                                                                           # load calories.csv
normalized_df_calories = pd.concat([df_calories.iloc[:,0:1],(df_calories.iloc[:,1:]-df_calories.iloc[:,1:].min())/(df_calories.iloc[:,1:].max()-df_calories.iloc[:,1:].min())],axis=1)          # calories normalization

df_merge = normalized_df_exercise.merge(normalized_df_calories, left_on='User_ID', right_on='User_ID').iloc[:,1:]                   # merge exercise 和 calories
df_merge_random = df_merge.reindex(np.random.permutation(df_merge.index))                                                           # random index
df_merge_random.reset_index(drop=True, inplace=True)                                                                                # reindex

df_training, df_validation, df_testing = np.split(df_merge_random, [int(.7*len(df_merge_random)), int(.8*len(df_merge_random))])    # split 70:10:20 for training, validation, and testing
df_training.reset_index(drop=True, inplace=True)                                                                                    # reindex
df_validation.reset_index(drop=True, inplace=True)                                                                                  # reindex
df_testing.reset_index(drop=True, inplace=True)                                                                                     # reindex

def gaussian_basis(X, mu, s):
    return np.exp(-((X-mu)**2 / (2*s**2)))                                          # calculate gaussian basis (Textbook 3.4)

def  Phi_matrix(X, X_mu, X_s):
    #print(X)
    #print(X_mu)
    #print(X_s)
    #print(X.shape)
    Phi = np.ones([X.shape[0], X.shape[1]])
    Phi[:, 0] = X[:, 0]
    for j in range(0, X.shape[1]):
        for i in range(0, X.shape[0]):
            Phi[i][j] = gaussian_basis(X[i][j], X_mu[j], X_s[j])                    # calculate design matrix, Phi (Textbook 3.16)
    return Phi

def BLR(train_data, train_label, test_data, train_mean, train_std):                 # functions BLR()
    train_Phi = Phi_matrix(train_data, train_mean, train_std)
    #print(train_Phi)
    weights = np.linalg.inv(np.identity(train_Phi.shape[1]) + train_Phi.T @ train_Phi) @ train_Phi.T @ train_label                  # calculate w (Textbook 3.28), lambda = 1
    y_pred = Phi_matrix(test_data, train_mean, train_std) @ weights                 # y_pred = Phi @ w (Textbook 3.31)
    return y_pred

def MLR(train_data, train_label, test_data, train_mean, train_std):                 # functions MLR()
    train_Phi = Phi_matrix(train_data, train_mean, train_std)
    #print(train_Phi)
    weights = np.linalg.inv(train_Phi.T @ train_Phi) @ train_Phi.T @ train_label    # calculate w (Textbook 3.15)
    y_pred = Phi_matrix(test_data, train_mean, train_std) @ weights                 # y_pred = Phi @ w (Textbook 3.31)
    return y_pred

select_feature = [0,1,2,3,4,5,6]                                                    # select features for training
train_mean = df_training.mean()[:-1][select_feature]                                # calculate mean of train data with those selected features
train_std = df_training.std()[:-1][select_feature]                                  # calculate std of train data with those selected features

train_data = df_training.values[:,select_feature]                                   # generate train_data with those selected features
train_label = df_training.values[:,-1]                                              # generate train_label
test_data = df_testing.values[:,select_feature]                                     # generate test_data with those selected features
test_label = df_testing.values[:,-1]                                                # generate test_label
validation_data = df_validation.values[:,select_feature]                            # generate validation_data with those selected features
validation_label = df_validation.values[:,-1]                                       # generate validation_label

# Q1:
y_pred_MLR = MLR(train_data, train_label, test_data, train_mean, train_std)         # predict test_data using MLR
mse_MLR = 1/len(test_data) * np.sum((test_label - y_pred_MLR)**2)                   # calculate mse of MLR
# Q2:
y_pred_BLR = BLR(train_data, train_label, validation_data, train_mean, train_std)   # predict validation_data using BLR
mse_BLR = 1/len(validation_data) * np.sum((validation_label - y_pred_BLR)**2)       # calculate mse of BLR

print('mse_BLR: ', mse_BLR, '\nmse_MLR: ', mse_MLR)

#----------------------------------------------------------------
'''
for i in range(0, df_merge.shape[1]-1):                                             
    plt.figure(figsize=(8, 8))
    plt.plot(df_merge.iloc[:,i].values, df_merge.iloc[:,-1].values, 'bo')           # plot all features vs Calories, to observe the usefulness of the feature
    plt.title(str(df_merge.columns[i]) + ' vs ' + str(df_merge.columns[-1]), size = 20)
    plt.xlabel(str(df_merge.columns[i]), size = 18)
    plt.ylabel(str(df_merge.columns[-1]), size = 18)
'''

#----------------------------------------------------------------
# Q3:
def MLR_coefficients(X, y):
    #co_MLR = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)
    coefs_MLR = np.linalg.inv(X.T @ X) @ X.T @ y                                    # calculate coffs of MLR, that is Intercept and Slope
    return coefs_MLR

def BLR_coefficients(X, y):
    coefs_BLR = np.linalg.inv(np.identity(X.shape[1]) + X.T @ X) @ X.T @ y          # calculate coffs of BLR, that is Intercept and Slope
    return coefs_BLR

df_training['intercept'] = 1
train_data_X = df_training.loc[:, ['intercept', 'Duration']]
train_label_y = df_training.loc[:, 'Calories']

coefs_MLR = MLR_coefficients(train_data_X, train_label_y)                           # generate MLR_coefficients of train data
print('MLR Intercept:', coefs_MLR[0])                                               # MLR Intercept
print('MLR Slope: ', coefs_MLR[1])                                                  # MLR Slope

coefs_BLR = BLR_coefficients(train_data_X, train_label_y)                           # generate BLR_coefficients of train data
print('BLR Intercept:', coefs_BLR[0])                                               # BLR Intercept
print('BLR Slope: ', coefs_BLR[1])                                                  # BLR Slope

xs = np.linspace(0, 1, 1000)                                                        # generate x croods
ys_MLR = coefs_MLR[0] + coefs_MLR[1] * xs                                           # calculate y croods of MLR
ys_BLR = coefs_BLR[0] + coefs_BLR[1] * xs                                           # calculate y croods of BLR

plt.figure(figsize=(8, 8))
plt.plot(df_training.loc[:,'Duration'], df_training.loc[:,'Calories'], 'bo', label = 'Observations', alpha = 0.8)                   # plot scatter
plt.xlabel('Duration', size = 18)
plt.ylabel('Calories', size = 18)
plt.plot(xs, ys_MLR, 'r', label = 'MLR Fit', linewidth = 3)                                                                         # plot regression line of MLR
plt.plot(xs, ys_BLR, 'b--', label = 'BLR Fit', linewidth = 3)                                                                       # plot regression line of BLR
plt.legend(prop={'size': 12})
plt.title('Duration vs Calories ', size = 20)

'''
with pm.Model() as linear_model:                                                    # linear model from pymc3
    # Intercept
    intercept = pm.Normal('Intercept', mu = 0, sd = 10)
    
    # Slope 
    slope = pm.Normal('slope', mu = 0, sd = 10)
    
    # Standard deviation
    sigma = pm.HalfNormal('sigma', sd = 10)
    
    # Estimate of mean
    mean = intercept + slope * X.loc[:, 'Duration']
    
    # Observed values
    Y_obs = pm.Normal('Y_obs', mu = mean, sd = sigma, observed = y.values)
    
    # Sampler
    step = pm.NUTS()

    # Posterior distribution
    linear_trace = pm.sample(1000, step)
    
    # https://github.com/WillKoehrsen/Data-Analysis/blob/master/bayesian_lr/Bayesian%20Linear%20Regression%20Demonstration.ipynb
'''

#----------------------------------------------------------------
# Q4:
lr = LinearRegression()                                                             # use LinearRegression model from sklearn
lr.fit(train_data, train_label)                                                     # fit model
pred = lr.predict(test_data)                                                        # predict test_data using LinearRegression model
mse_lr_test = 1/len(test_data) * np.sum((test_label - pred)**2)                     # calculate mse of LinearRegression model
pred = lr.predict(validation_data)                                                  # predict validation_data using LinearRegression model
mse_lr_validation= 1/len(validation_data) * np.sum((validation_label - pred)**2)    # calculate mse of LinearRegression model
print('mse_lr_validation: ', mse_lr_validation, '\nmse_lr_test: ', mse_lr_test)
import pandas as pd
import openpyxl
import numpy as np
import scipy.stats as st
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
#from sklearn.linear_model import LogisticRegression

#---------------------------------------------------------
#Import Regressors from Libraries
#---------------------------------------------------------
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet

#--------------------------------------------------------
# Import model metrics from libraries
#--------------------------------------------------------
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error

#--------------------------------------------------------
#Reading file
#--------------------------------------------------------
features = pd.read_csv('feature.csv')
features.head()
features.info()
features.columns

#--------------------------------------------------------
#x variable
#--------------------------------------------------------
feature_cols = [
                'light_physical_activity',
                'age',
                ' time_sitting ',
                'consistency_of_interest',
                'reaps_score',
                'total_mood_disturbance',
                'poms_fatigue',
                'poms_depression',
                'moderate_physical_activity',
                'mvpa',
                'intensity_mental_work_work_school days',
                'intensity_mental_work_non-school_work days']

X = features[feature_cols]

#-------------------------------------------------------
#y variable
#-------------------------------------------------------
y = features.total_surveys

#-------------------------------------------------------
# Test and Train separation
#-------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

#-------------------------------------------------------
# Regressor Models
#-------------------------------------------------------
#models = []

'''
DecisionTreeRegressorModel =  DecisionTreeRegressor(random_state=33)
LinearRegressionModel = LinearRegression(fit_intercept=True, normalize=True,copy_X=True,n_jobs=-1)
RandomForestRegressorModel = RandomForestRegressor(random_state=33)
RidgeRegressionModel = Ridge(alpha=1.0, random_state=33)
LassoRegressionModel = Lasso(alpha=1.0, random_state=33,normalize=False)
SGDRegressionModel = SGDRegressor(alpha=0.1, random_state=33,penalty='l2',loss = 'huber')
MLPRegressorModel = MLPRegressor(activation='tanh',solver='lbfgs',learning_rate='constant',early_stopping= False,alpha=0.0001 ,hidden_layer_sizes=(100, 3),random_state=33)
SVRModel = SVR(C = 1.0 ,epsilon=0.1,kernel = 'rbf')
GBRModel = GradientBoostingRegressor(learning_rate = 1.5 ,random_state=33)
NeighborsRegressorModel = KNeighborsRegressor(n_neighbors = 5, weights='uniform',algorithm = 'auto')
ElasticNetModel = ElasticNet(random_state=0)
'''


# prepare the cross-validation procedure
cv = KFold(n_splits=10, random_state=1, shuffle=True)
# create model
model = SVR(C = 1.0 ,epsilon=0.1,kernel = 'rbf')
model.fit(X_train, y_train)
# evaluate model
scores = []
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

print(scores)
# report performance
#print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

#model.fit(X_train, y_train)
ypred = model.predict(X_test)
mse = mean_squared_error(y_test,ypred)

r2_score = scores

print(
    "ypred:", ypred, '\n',
    "r2_score:", scores, '\n',
    "Mean_r2:", np.average(r2_score), '\n',
    "CI:", st.t.interval(alpha=0.95, df=len(r2_score)-1, loc=np.mean(r2_score), scale=st.sem(r2_score)), '\n',
    "Min_r2:", np.min(r2_score), '\n',
    "Q1_r2:", np.quantile(r2_score,0.25), '\n',
    "Q2_r2:", np.quantile(r2_score,0.50), '\n',
    "Q3_r2:", np.quantile(r2_score,0.75), '\n',
    "Max_r2:", np.max(r2_score), '\n',
    "mse:", mse, '\n',
    "mse_mean:", np.average(mse), '\n',
    "mse_STD:", np.std(mse)*2, '\n',
    "mae: ", metrics.mean_absolute_error(y_test,ypred)
    )



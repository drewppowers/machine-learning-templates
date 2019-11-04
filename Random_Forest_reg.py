# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('seaborn-deep')
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Reading in data
df = pd.read_csv("FILE_NAME_HERE.csv")

#Change to appropriate
X = df.iloc[:,].values
y = df.iloc[:,4].values


# Splitting and scaling 
X_train, X_test, y_train, y_test = train_test_split(X,y)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Fitting the Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 100, oob_score=True)
regressor.fit(X_train,y_train)


# Grid Search
# Random Forest Regressor parameters 
parameters = [ 
    {'n_estimators' : [2**i for i in range(2,10)], 'criterion' : ['mse', 'mae'],
     'max_depth' : [2**i for i in range(1,7)], 'min_samples_split' : [2**i for i in range(1,5)],
     'min_samples_leaf' : [2**i for i in range(1,5)], 'max_features' : ['auto', 'sqrt', 'log2'],
     'min_impurity_decrease' : [.5**i for i in range(1,8)],  'n_jobs' : [-1],
                }]
grid_search = GridSearchCV(regressor, parameters, scoring='accuracy', cv=10,
                           n_jobs=-1)

grid_search = grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(grid_search.best_score_)


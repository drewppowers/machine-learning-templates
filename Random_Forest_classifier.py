import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

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

## Classifier

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators =  10, criterion = 'entropy')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Grid Search
# Random Forest Classifier parameters 
parameters = [ 
    {'n_estimators' : [2**i for i in range(2,10)], 'criterion' : ['mse', 'mae'],
     'max_depth' : [2**i for i in range(1,7)], 'min_samples_split' : [2**i for i in range(1,5)],
     'min_samples_leaf' : [2**i for i in range(1,5)], 'max_features' : ['auto', 'sqrt', 'log2'],
     'min_impurity_decrease' : [.5**i for i in range(1,8)],  'n_jobs' : [-1],
     'class_weight': ['balanced', None]
                }]

grid_search = GridSearchCV(clf, parameters, scoring='accuracy', cv=10,
                           n_jobs=-1)

grid_search = grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print(grid_search.best_score_)
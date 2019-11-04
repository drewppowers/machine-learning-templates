import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Reading in data
df = pd.read_csv("FILE_NAME_HERE.csv")

# drop objects (Alternatively, dummy vars)
df = df.select_dtypes(exclude=['object'])

#Change to appropriate
X = df.iloc[:,].values
y = df.iloc[:,4].values

# Splitting and scaling 
X_train, X_test, y_train, y_test = train_test_split(X,y)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Model 
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
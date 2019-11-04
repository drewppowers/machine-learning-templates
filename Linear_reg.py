import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')

# Reading in data
df = pd.read_csv("FILE_NAME_HERE.csv")

#Change to appropriate
X = df.iloc[:,].values
y = df.iloc[:,4].values

# drop objects
df = df.select_dtypes(exclude=['object'])

# Training/testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.35,
         random_state=1234)

# Regressor
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

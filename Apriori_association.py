import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')

# Reading in data
df = pd.read_csv('FILE_NAME_HERE.csv', header = None)
df = df.values

# Creating the trans list of lists
trans = []

for i in range(df.shape[0]):
    trans.append([str(df[i][j]) for j in range(df.shape[1])])

# Training Apriori on the dataset
from apyori import apriori
rules = apriori(trans, min_support = 0.003, min_confidence = 0.2, 
                min_lift = 5, min_length = 2)


# Visualising the results
results = list(rules)
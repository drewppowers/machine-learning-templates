import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
from sklearn.cluster import KMeans
import matplotlib.cm
cmap = matplotlib.cm.get_cmap('plasma')

# Reading in data
df = pd.read_csv("FILE_NAME_HERE.csv")

#Change to appropriate. No y here. 
X = df.iloc[:,].values

# Choosing the value of k by the elbow method
wcss = []

for i in range(1,21):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit_transform(X)
    wcss.append(kmeans.inertia_)
    
plt.figure()
plt.plot(range(1,21), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Clustering the data

#Change k to appropriate value from elbow method
k = 5
kmeans = KMeans(n_clusters = k)
y_kmeans = kmeans.fit_predict(X)

labels = [('Cluster ' + str(i+1)) for i in range(k)]

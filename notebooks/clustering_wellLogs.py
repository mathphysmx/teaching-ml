import pandas as pd
import lasio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_samples, silhouette_score

las = lasio.read("datasets/47-019-00241-00-00.txt")
las.keys()
las['GR']

df = las.df()
df = df.iloc[1:, ]

gr = pd.to_numeric(df['GR'])
gr.replace(to_replace=-999.25, value=np.nan, inplace=True)
gr.dropna(inplace=True)
gr.describe()

kmeans = KMeans(n_clusters=5)
kmeans.fit(gr.values.reshape(-1,1))
kmeans.cluster_centers_
kmeans.labels_
result = pd.DataFrame({'GR':gr, 'Lithology':kmeans.labels_})
# result.to_excel('clusters_GR.xlsx')
graph = result.plot(y='GR')
# for i in kmeans.cluster_centers_:
#     graph.axhline(i, c = 'g-')
## CUTOFF
centers = np.sort(kmeans.cluster_centers_[:,0])
n = (centers.shape[0]-1)
print(centers[0:n] + np.diff(centers)/2) # cutoff
for cutoff in (centers[0:n] + np.diff(centers)/2):
    graph.axhline(cutoff, c = 'g')
plt.show()

result.reset_index(inplace=True)
graph = sns.scatterplot(x="DEPT", y="GR", data=result, hue='Lithology', palette="Set2")
plt.show()

########## KMEANS MULTIVARIATE

import pandas as pd
import lasio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances

las = lasio.read("datasets/47-019-00241-00-00.txt")
las.keys()
df = las.df().iloc[1:,:].apply(pd.to_numeric).replace(-999.25, np.nan)
df.dropna(inplace=True)

# df.GR.plot()
# plt.show()

kmeans = KMeans(n_clusters=2).fit(df.values)
df['Lithology'] = kmeans.labels_
graph = df.GR.plot(c = kmeans.labels_)
df.reset_index(inplace=True)
graph = sns.scatterplot(x="DEPT", y="GR", data=df, hue='Lithology', palette="Set2")
# https://resinsight.org/plot-window/welllogsandplots/  
plt.show()

silhouette_avg = silhouette_score(df, kmeans.labels_)

# Compute the silhouette scores for each sample
sample_silhouette_values = silhouette_samples(df, kmeans.labels_)



### WELL LOG MANIPULATION
import numpy as np
import matplotlib.pyplot as plt
import welly
welly.__version__
from welly import Well
w = Well.from_las("datasets/P-129_out.LAS")
w.data['GR']
w.plot()
plt.show()
gr = w.data['GR']
gr[1000:1010]
gr.plot(); plt.show()
gr.plot(lw=0.3); plt.show()
gr.plot_2d(cmap='viridis', curve=True, lw=0.3, edgecolor='k')
plt.show()



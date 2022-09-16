# standard DS imports
import pandas as pd
import numpy as np

# viz and stats
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# scaling and modeling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

def run_kmeans(train_scaled, max_centroids=15):
    '''
    This function takes in the scaled train data set (continuous features only)
    and the max number of centroids desired.
    
    Outputs the seaborn plot of centroids vs inertia to visualize the 'elbow' method.
    '''
    
    n = 1
    points = {}
    while n <= max_centroids:
        km = KMeans(n_clusters = n)
        km.fit(train_scaled)
        points[f'km_{n}'] = {'centroids':n, 'inertia': km.inertia_}
        n+=1
    
    points = pd.DataFrame(points).T
    
    sns.relplot(data=points, x='centroids', y='inertia').set(title='Elbow Method Plot')
    # x = range(0,40,1)
    # y = range(0,40,1)
    # plt.plot(x,y)
    # plt.xlim(0)
    # plt.ylim(0)
    plt.grid()
    plt.show()
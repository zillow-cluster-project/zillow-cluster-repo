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
    
    
def get_models_with_results(X_train, y_train, X_val, y_val):
    ''' 
    This function takes in the X and y objects and then runs the following models:
    - Baseline model using y_train mean
    - LarsLasso model with alpha=1
    - Quadratic Linear Regression
    - Cubic Linear Regression
    
    Returns a DataFrame with the results.
    '''
    # Baseline Model
    # run the model
    pred_mean = y_train.value.mean()
    y_train['pred_mean'] = pred_mean
    y_val['pred_mean'] = pred_mean
    rmse_train = mean_squared_error(y_train.value, y_train.pred_mean, squared=False)
    rmse_val = mean_squared_error(y_val.value, y_val.pred_mean, squared=False)

    # save the results
    metrics = pd.DataFrame(data=[{
        'model': 'baseline_mean',
        'rmse_train': rmse_train,
        'r2_train': explained_variance_score(y_train.value, y_train.pred_mean),
        'rmse_val': rmse_val,
        'r2_val': explained_variance_score(y_val.value, y_val.pred_mean)}])

    # LassoLars Model
    # run the model
    lars = LassoLars(alpha=1)
    lars.fit(X_train, y_train.value)
    y_train['pred_lars'] = lars.predict(X_train)
    rmse_train = mean_squared_error(y_train.value, y_train.pred_lars, squared=False)
    y_val['pred_lars'] = lars.predict(X_val)
    rmse_val = mean_squared_error(y_val.value, y_val.pred_lars, squared=False)

    # save the results
    metrics = metrics.append({
        'model': 'LarsLasso, alpha 1',
        'rmse_train': rmse_train,
        'r2_train': explained_variance_score(y_train.value, y_train.pred_lars),
        'rmse_val': rmse_val,
        'r2_val': explained_variance_score(y_val.value, y_val.pred_lars)}, ignore_index=True)

    # Polynomial Models
    # set up the model
    pf = PolynomialFeatures(degree=2)
    X_train_d2 = pf.fit_transform(X_train)
    X_val_d2 = pf.transform(X_val)
    
    # run the model
    lm2 = LinearRegression()
    lm2.fit(X_train_d2, y_train.value)
    y_train['pred_lm2'] = lm2.predict(X_train_d2)
    rmse_train = mean_squared_error(y_train.value, y_train.pred_lm2, squared=False)
    y_val['pred_lm2'] = lm2.predict(X_val_d2)
    rmse_val = mean_squared_error(y_val.value, y_val.pred_lm2, squared=False)

    # save the results
    metrics = metrics.append({
        'model': 'Quadratic Linear Regression',
        'rmse_train': rmse_train,
        'r2_train': explained_variance_score(y_train.value, y_train.pred_lm2),
        'rmse_val': rmse_val,
        'r2_val': explained_variance_score(y_val.value, y_val.pred_lm2)}, ignore_index=True)

    # set up the model
    pf = PolynomialFeatures(degree=3)
    X_train_d2 = pf.fit_transform(X_train)
    X_val_d2 = pf.transform(X_val)

    # run the model
    lm2 = LinearRegression()
    lm2.fit(X_train_d2, y_train.value)
    y_train['pred_lm2'] = lm2.predict(X_train_d2)
    rmse_train = mean_squared_error(y_train.value, y_train.pred_lm2, squared=False)
    y_val['pred_lm2'] = lm2.predict(X_val_d2)
    rmse_val = mean_squared_error(y_val.value, y_val.pred_lm2, squared=False)

    # save the results
    metrics = metrics.append({
        'model': 'Cubic Linear Regression',
        'rmse_train': rmse_train,
        'r2_train': explained_variance_score(y_train.value, y_train.pred_lm2),
        'rmse_val': rmse_val,
        'r2_val': explained_variance_score(y_val.value, y_val.pred_lm2)}, ignore_index=True)

    return metrics
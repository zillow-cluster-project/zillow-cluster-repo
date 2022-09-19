# standard DS imports
import pandas as pd
import numpy as np

# for feature selection verification and evaluation 
from sklearn.metrics import r2_score, mean_squared_error, explained_variance_score
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler



def get_models(train, val, test):

    cols_to_scale = ['baths', 'beds', 'living_space', 'county', 'lat', 'long', 'lotsize', 'pool', 'city_id','year_built', 'tax_value', 'price_sqft',
                    'mvp_0', 'mvp_1', 'mvp_2', 'mvp_3', 
                    'value_0', 'value_1', 'value_2',
                    'size_0', 'size_1', 'size_2']

    train_scaled = train.copy()
    val_scaled = val.copy()
    test_scaled = test.copy()

    # create the scaler and fit it
    scaler = MinMaxScaler()
    scaler.fit(train[cols_to_scale])

    # use the scaler to scale the data and resave
    train_scaled[cols_to_scale] = pd.DataFrame(scaler.transform(train[cols_to_scale]),
                                               columns = train[cols_to_scale].columns.values).set_index([train.index.values])
    val_scaled[cols_to_scale] = pd.DataFrame(scaler.transform(val[cols_to_scale]),
                                               columns = val[cols_to_scale].columns.values).set_index([val.index.values])
    test_scaled[cols_to_scale] = pd.DataFrame(scaler.transform(test[cols_to_scale]),
                                               columns = test[cols_to_scale].columns.values).set_index([test.index.values])
    
    
    X_train, y_train = train_scaled[['baths', 'beds', 'living_space', 'lat', 'long', 'county', 'lotsize', 'year_built', 'tax_value', 'price_sqft',
                    'mvp_0', 'mvp_2', 
                    # 'value_0', 'value_1', 
                                     'value_2',
                    # 'size_0', 'size_1', 'size_2'
                                    ]], train.logerror
    X_val, y_val = val_scaled[['baths', 'beds', 'living_space', 'lat', 'long', 'county', 'lotsize', 'year_built', 'tax_value', 'price_sqft',
                    'mvp_0', 'mvp_2',  
                    # 'value_0', 'value_1', 
                               'value_2',
                    # 'size_0', 'size_1', 'size_2'
                              ]], val.logerror
    X_test, y_test = test_scaled[['baths', 'beds', 'living_space', 'lat', 'long', 'county', 'lotsize', 'year_built', 'tax_value', 'price_sqft',
                    'mvp_0', 'mvp_2', 
                    # 'value_0', 'value_1', 
                                  'value_2',
                    # 'size_0', 'size_1', 'size_2'
                                 ]], test.logerror
    y_train = pd.DataFrame(y_train)
    y_val = pd.DataFrame(y_val)
    y_test = pd.DataFrame(y_test)


    # Baseline Model
    # run the model
    pred_mean = y_train.logerror.mean()
    y_train['pred_mean'] = pred_mean
    y_val['pred_mean'] = pred_mean
    y_test['pred_mean'] = pred_mean
    rmse_train = mean_squared_error(y_train.logerror, y_train.pred_mean, squared=False)
    rmse_val = mean_squared_error(y_val.logerror, y_val.pred_mean, squared=False)

    # save the results
    metrics = pd.DataFrame(data=[{
        'model': 'baseline_mean',
        'rmse_train': rmse_train,
        'r2_train': explained_variance_score(y_train.logerror, y_train.pred_mean),
        'rmse_val': rmse_val,
        'r2_val': explained_variance_score(y_val.logerror, y_val.pred_mean)
    }])


    lm = LinearRegression()
    lm.fit(X_train, y_train.logerror)
    y_train['pred_lm'] = lm.predict(X_train)
    rmse_train = mean_squared_error(y_train.logerror, y_train.pred_lm, squared=False)
    y_val['pred_lm'] = lm.predict(X_val)
    rmse_val = mean_squared_error(y_val.logerror, y_val.pred_lm, squared=False)
    # print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train, 
    #       "\nValidation/Out-of-Sample: ", rmse_val)

    metrics = metrics.append({
        'model': 'OLS',
        'rmse_train': rmse_train,
        'r2_train': explained_variance_score(y_train.logerror, y_train.pred_lm),
        'rmse_val': rmse_val,
        'r2_val': explained_variance_score(y_val.logerror, y_val.pred_lm)
        }, ignore_index=True)

    # LassoLars Model
    # run the model
    lars = LassoLars(alpha=1)
    lars.fit(X_train, y_train.logerror)
    y_train['pred_lars'] = lars.predict(X_train)
    rmse_train = mean_squared_error(y_train.logerror, y_train.pred_lars, squared=False)
    y_val['pred_lars'] = lars.predict(X_val)
    rmse_val = mean_squared_error(y_val.logerror, y_val.pred_lars, squared=False)

    # save the results
    metrics = metrics.append({
        'model': 'LarsLasso, alpha 1',
        'rmse_train': rmse_train,
        'r2_train': explained_variance_score(y_train.logerror, y_train.pred_lars),
        'rmse_val': rmse_val,
        'r2_val': explained_variance_score(y_val.logerror, y_val.pred_lars
        )}, ignore_index=True)


    # Polynomial Models
    # set up the model
    pf = PolynomialFeatures(degree=2)
    X_train_d2 = pf.fit_transform(X_train)
    X_val_d2 = pf.transform(X_val)

    # run the model
    lm2 = LinearRegression()
    lm2.fit(X_train_d2, y_train.logerror)
    y_train['pred_lm2'] = lm2.predict(X_train_d2)
    rmse_train = mean_squared_error(y_train.logerror, y_train.pred_lm2, squared=False)
    y_val['pred_lm2'] = lm2.predict(X_val_d2)
    rmse_val = mean_squared_error(y_val.logerror, y_val.pred_lm2, squared=False)

    # save the results
    metrics = metrics.append({
        'model': 'Quadratic Linear Regression',
        'rmse_train': rmse_train,
        'r2_train': explained_variance_score(y_train.logerror, y_train.pred_lm2),
        'rmse_val': rmse_val,
        'r2_val': explained_variance_score(y_val.logerror, y_val.pred_lm2)
    }, ignore_index=True)

    # set up the model
    pf = PolynomialFeatures(degree=3)
    X_train_d2 = pf.fit_transform(X_train)
    X_val_d2 = pf.transform(X_val)

    # run the model
    lm2 = LinearRegression()
    lm2.fit(X_train_d2, y_train.logerror)
    y_train['pred_lm2'] = lm2.predict(X_train_d2)
    rmse_train = mean_squared_error(y_train.logerror, y_train.pred_lm2, squared=False)
    y_val['pred_lm2'] = lm2.predict(X_val_d2)
    rmse_val = mean_squared_error(y_val.logerror, y_val.pred_lm2, squared=False)

    # save the results
    metrics = metrics.append({
        'model': 'Cubic Linear Regression',
        'rmse_train': rmse_train,
        'r2_train': explained_variance_score(y_train.logerror, y_train.pred_lm2),
        'rmse_val': rmse_val,
        'r2_val': explained_variance_score(y_val.logerror, y_val.pred_lm2)
    }, ignore_index=True)

    return metrics, X_train, y_train, X_val, y_val, X_test, y_test


def run_test(X_train, y_train, X_val, y_val, X_test, y_test):
    # run the model
    lm = LinearRegression()
    lm.fit(X_train, y_train.logerror)
    y_train['pred_lm'] = lm.predict(X_train)
    rmse_train = mean_squared_error(y_train.logerror, y_train.pred_lm, squared=False)
    y_val['pred_lm'] = lm.predict(X_val)
    rmse_val = mean_squared_error(y_val.logerror, y_val.pred_lm, squared=False)
    y_test['pred_lm'] = lm.predict(X_test)
    rmse_test = mean_squared_error(y_test.logerror, y_test.pred_lm, squared=False)


    # save the results
    results = pd.DataFrame({'train': 
                               {
                                'rmse': rmse_train, 
                                'r2': explained_variance_score(y_train.logerror, y_train.pred_lm)},
                           'validate': 
                               {
                                'rmse': rmse_val, 
                                'r2': explained_variance_score(y_val.logerror, y_val.pred_lm)},
                           'test': 
                               {
                                'rmse': rmse_test, 
                                'r2': explained_variance_score(y_test.logerror, y_test.pred_lm)}
                          })
    return results.T
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import wrangle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

def make_clusters(train, val, test):
    # Make mvp clusters
    cols = ['baths','beds','living_space', 'city_id','tax_value', 'year_built', 'price_sqft']
    train_scaled, val_scaled, test_scaled = wrangle.scale_data(train, val, test, cols)
    
    km = KMeans(n_clusters = 4)
    km.fit(train_scaled[['baths','beds','living_space', 'city_id','tax_value', 'year_built']])
    train['mvp_cluster'] = km.predict(train_scaled[['baths','beds','living_space', 'city_id','tax_value', 'year_built']])
    val['mvp_cluster'] = km.predict(val_scaled[['baths','beds','living_space', 'city_id','tax_value', 'year_built']])
    test['mvp_cluster'] = km.predict(test_scaled[['baths','beds','living_space', 'city_id','tax_value', 'year_built']])
    
    # Make value clusters
    cols = ['structure_value','tax_value', 'land_value', 'taxes', 'price_sqft']
    train_scaled, val_scaled, test_scaled = wrangle.scale_data(train, val, test, cols)

    km = KMeans(n_clusters = 3)
    km.fit(train_scaled[['structure_value','tax_value', 'land_value', 'taxes', 'price_sqft']])
    train['value_cluster'] = km.predict(train_scaled[['structure_value','tax_value', 'land_value', 'taxes', 'price_sqft']])
    val['value_cluster'] = km.predict(val_scaled[['structure_value','tax_value', 'land_value', 'taxes', 'price_sqft']])
    test['value_cluster'] = km.predict(test_scaled[['structure_value','tax_value', 'land_value', 'taxes', 'price_sqft']])
    
    # Make size clusters
    train_scaled, val_scaled, test_scaled = wrangle.scale_data(train, val, test, cols)

    km = KMeans(n_clusters = 3)
    km.fit(train_scaled[['basement_sqft','area', 'area12', 'lotsize','living_space']])
    train['size_cluster'] = km.predict(train_scaled[['basement_sqft','area', 'area12', 'lotsize','living_space']])
    val['size_cluster'] = km.predict(val_scaled[['basement_sqft','area', 'area12', 'lotsize','living_space']])
    test['size_cluster'] = km.predict(test_scaled[['basement_sqft','area', 'area12', 'lotsize','living_space']])
    
    
    
    dummies = pd.get_dummies(train.mvp_cluster, columns=['mvp_0', 'mvp_1', 'mvp_2', 'mvp_3'])
    dummie = pd.get_dummies(train.value_cluster, columns=['value_0', 'value_1', 'value_2'])
    dum = pd.get_dummies(train.size_cluster, columns=['size_0', 'size_1', 'size_2'])
    train = pd.concat([train, dummies, dummie, dum], axis=1)
    
    dummies = pd.get_dummies(val.mvp_cluster, columns=['mvp_0', 'mvp_1', 'mvp_2', 'mvp_3'])
    dummie = pd.get_dummies(val.value_cluster, columns=['value_0', 'value_1', 'value_2'])
    dum = pd.get_dummies(val.size_cluster, columns=['size_0', 'size_1', 'size_2'])
    val = pd.concat([val, dummies, dummie, dum], axis=1)
    val['size_1'] = 0
    
    dummies = pd.get_dummies(test.mvp_cluster, columns=['mvp_0', 'mvp_1', 'mvp_2', 'mvp_3'])
    dummie = pd.get_dummies(test.value_cluster, columns=['value_0', 'value_1', 'value_2'])
    dum = pd.get_dummies(test.size_cluster, columns=['size_0', 'size_1', 'size_2'])
    test = pd.concat([test, dummies, dummie, dum], axis=1)
    
    
    train.columns = ['basement_sqft', 'baths', 'beds', 'bathnbed', 'decktype', 'area', 'area12', 'county', 'fireplace', 'fullbath',
                     'hottub_or_spa', 'lat', 'long', 'lotsize', 'pool', 'pool10', 'pool2', 'pool7', 'landuse_code', 'raw_census',
                     'city_id', 'county_id', 'zip_id', 'rooms', 'threequarterbnb', 'year_built', 'fireplace_flag', 'structure_value',
                     'tax_value', 'assessment_year', 'land_value', 'taxes', 'tax_delq_flag', 'tax_delq_year', 'census', 'logerror',
                     'transactiondate', 'construction_type', 'landuse_desc', 'living_space', 'price_sqft', 'mvp_cluster', 'value_cluster',
                     'size_cluster', 'mvp_0', 'mvp_1', 'mvp_2', 'mvp_3', 'value_0', 'value_1', 'value_2', 'size_0', 'size_1', 'size_2']
    
    val.columns = ['basement_sqft', 'baths', 'beds', 'bathnbed', 'decktype', 'area', 'area12', 'county', 'fireplace', 'fullbath',
                     'hottub_or_spa', 'lat', 'long', 'lotsize', 'pool', 'pool10', 'pool2', 'pool7', 'landuse_code', 'raw_census',
                     'city_id', 'county_id', 'zip_id', 'rooms', 'threequarterbnb', 'year_built', 'fireplace_flag', 'structure_value',
                     'tax_value', 'assessment_year', 'land_value', 'taxes', 'tax_delq_flag', 'tax_delq_year', 'census', 'logerror',
                     'transactiondate', 'construction_type', 'landuse_desc', 'living_space', 'price_sqft', 'mvp_cluster', 'value_cluster',
                     'size_cluster', 'mvp_0', 'mvp_1', 'mvp_2', 'mvp_3', 'value_0', 'value_1', 'value_2', 'size_0', 'size_2', 'size_1']
    
    val = val[['basement_sqft', 'baths', 'beds', 'bathnbed', 'decktype', 'area', 'area12', 'county', 'fireplace', 'fullbath',
                     'hottub_or_spa', 'lat', 'long', 'lotsize', 'pool', 'pool10', 'pool2', 'pool7', 'landuse_code', 'raw_census',
                     'city_id', 'county_id', 'zip_id', 'rooms', 'threequarterbnb', 'year_built', 'fireplace_flag', 'structure_value',
                     'tax_value', 'assessment_year', 'land_value', 'taxes', 'tax_delq_flag', 'tax_delq_year', 'census', 'logerror',
                     'transactiondate', 'construction_type', 'landuse_desc', 'living_space', 'price_sqft', 'mvp_cluster', 'value_cluster',
                     'size_cluster', 'mvp_0', 'mvp_1', 'mvp_2', 'mvp_3', 'value_0', 'value_1', 'value_2', 'size_0', 'size_1', 'size_2']]
    
    test.columns = ['basement_sqft', 'baths', 'beds', 'bathnbed', 'decktype', 'area', 'area12', 'county', 'fireplace', 'fullbath',
                     'hottub_or_spa', 'lat', 'long', 'lotsize', 'pool', 'pool10', 'pool2', 'pool7', 'landuse_code', 'raw_census',
                     'city_id', 'county_id', 'zip_id', 'rooms', 'threequarterbnb', 'year_built', 'fireplace_flag', 'structure_value',
                     'tax_value', 'assessment_year', 'land_value', 'taxes', 'tax_delq_flag', 'tax_delq_year', 'census', 'logerror',
                     'transactiondate', 'construction_type', 'landuse_desc', 'living_space', 'price_sqft', 'mvp_cluster', 'value_cluster',
                     'size_cluster', 'mvp_0', 'mvp_1', 'mvp_2', 'mvp_3', 'value_0', 'value_1', 'value_2', 'size_0', 'size_1', 'size_2']
    
    return train, val, test
    
    
    
    
    
    
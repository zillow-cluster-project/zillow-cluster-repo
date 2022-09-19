import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os
import env
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



def acquire_zillow():
    '''
    This function checks if the zillow data is saved locally. 
    If it is not local, this function reads the zillow data from 
    the CodeUp MySQL database and return it in a DataFrame.
    '''
    
    # Acquire
    # Set file name
    filename = 'zillow.csv'
    # if the file is saved locally... grab that
    if os.path.isfile(filename):
        df = pd.read_csv(filename).iloc[:,1:]
    # if the file is not local, pull it via SQL from the CodeUp database
    else:
        q = '''SELECT *
                FROM properties_2017
                LEFT JOIN predictions_2017
                    USING (parcelid)
                LEFT JOIN heatingorsystemtype
                    USING (heatingorsystemtypeid)
                LEFT JOIN buildingclasstype
                    USING (buildingclasstypeid)
                LEFT JOIN architecturalstyletype
                    USING (architecturalstyletypeid)
                LEFT JOIN airconditioningtype
                    USING (airconditioningtypeid)
                LEFT JOIN storytype
                    USING (storytypeid)
                LEFT JOIN typeconstructiontype
                    USING (typeconstructiontypeid)
                LEFT JOIN propertylandusetype
                    USING (propertylandusetypeid)
                WHERE transactiondate LIKE '2017%%'
                AND latitude is not NULL
                AND longitude is not NULL
                AND (propertylandusetypeid = 261 OR propertylandusetypeid = 279);
                '''
                
        df = pd.read_sql(q, env.conn('zillow'))
        # only keep the most recent transaction for each home
        df = df.sort_values('transactiondate').drop_duplicates(keep='last')
        # drop the foreign keys and other unnecessary columns
        df = df.drop(columns=['heatingorsystemtypeid', 
                              'buildingclasstypeid',
                              'architecturalstyletypeid', 
                              'airconditioningtypeid',
                              'storytypeid',
                              'typeconstructiontypeid',
                              'propertylandusetypeid',
                              'parcelid',
                              'finishedsquarefeet13',
                              'finishedsquarefeet15',
                              'buildingclassdesc'])
        
        # Save it locally for future use
        df.to_csv(filename)
        df = pd.read_csv(filename).iloc[:,1:]
    # return the file
    return df


def prepare_zillow(df):
    '''
    This function takes in the zillow dataframe after it has been acquired
    from the acquire_zillow function. It drops columns, handles nulls,
    renames features, and feature engineers before returning the complete
    dataframe.
    '''
    # drop foreign key columns or ones that aren't needed
    df = df.drop(columns=['buildingqualitytypeid','propertyzoningdesc','unitcnt','heatingorsystemdesc','id','id.1'])
    
    # fill missing values
    df.fullbathcnt = df.fullbathcnt.fillna(0)
    df.pooltypeid2 = df.pooltypeid2.fillna(0)
    df.pooltypeid10 = df.pooltypeid10.fillna(0)
    df.pooltypeid7 = df.pooltypeid7.fillna(0)
    df.fireplacecnt = df.fireplacecnt.fillna(0)
    df.decktypeid = df.decktypeid.fillna(0)
    df.poolcnt = df.poolcnt.fillna(0)
    df.hashottuborspa = df.hashottuborspa.fillna(0)
    df.typeconstructiondesc = df.typeconstructiondesc.fillna('None')
    df.fireplaceflag = df.fireplaceflag.fillna(0)
    df.threequarterbathnbr = df.threequarterbathnbr.fillna(0)
    df.taxdelinquencyyear = df.taxdelinquencyyear.fillna(99999)
    df.taxdelinquencyflag = df.taxdelinquencyflag.fillna('N')
    df.calculatedbathnbr = df.calculatedbathnbr.fillna(0)
    df.basementsqft = df.basementsqft.fillna(0)
    df.numberofstories.value_counts(dropna=False)
    
    # drop columns or indicies beyond the threshold of nulls
    df = handle_missing_values(df, prop_req_cols=.6, prop_req_rows=.75)
    
    # drop any remaining indicies with nulls
    df = df.dropna()
    # drop the weird zipcode
    df = df.drop(index=df[df.regionidzip == df.regionidzip.max()].index.tolist())
    # rename all the columns
    df.columns = ['basement_sqft', 'baths', 'beds', 'bathnbed', 'decktype', 'area', 'area12', 'county', 'fireplace', 'fullbath', 'hottub_or_spa', 'lat', 'long', 'lotsize',
              'pool', 'pool10', 'pool2', 'pool7', 'landuse_code', 'raw_census', 'city_id', 'county_id', 'zip_id', 'rooms', 'threequarterbnb', 'year_built',
               'fireplace_flag', 'structure_value', 'tax_value', 'assessment_year', 'land_value', 'taxes', 'tax_delq_flag', 'tax_delq_year', 'census', 'logerror',
               'transactiondate', 'construction_type', 'landuse_desc']
    # do some feature engineering
    df['living_space'] = df.area - df.baths*60 - df.beds*200
    df['price_sqft'] = df.tax_value/df.area
    
    return df


def handle_missing_values(df, prop_req_cols=0.5, prop_req_rows=0.75):
    '''
    This function takes in a dataframe and drops columns with more than 50%
    missing values and rows with more than 75% missing values before returning
    the DataFrame. 
    '''
    threshold = int(round(prop_req_cols * len(df.index), 0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_req_rows * len(df.columns), 0))
    df = df.dropna(axis=0, thresh=threshold)
    
    return df


def split_data(df):
    '''
    Takes in a dataframe and target (as a string). Returns train, validate, and test subset 
    dataframes with the .2/.8 and .25/.75 splits to create a final .2/.2/.6 split between datasets
    '''
    # split the data into train and test. 
    train, test = train_test_split(df, test_size = .2, random_state=123)
    
    # split the train data into train and validate
    train, validate = train_test_split(train, test_size = .25, random_state=123)
    
    return train, validate, test


def scale_data(train, val, test, cols_to_scale):
    '''
    This function takes in train, validate, and test dataframes as well as a
    list of features to be scaled via the MinMaxScalar. It then returns the 
    scaled versions of train, validate, and test in new dataframes. 
    '''
    # create copies to not mess with the original dataframes
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
    
    return train_scaled, val_scaled, test_scaled
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
    '''
    This function takes in train, validate, and test dataframes, creates the mvp, 
    value, and size clusters, turns them into dummies, and renames the features to
    be human readable before returning all three dataframes. 
    '''
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
    
    
    # Make cluster dummies for train
    dummies = pd.get_dummies(train.mvp_cluster, columns=['mvp_0', 'mvp_1', 'mvp_2', 'mvp_3'])
    dummie = pd.get_dummies(train.value_cluster, columns=['value_0', 'value_1', 'value_2'])
    dum = pd.get_dummies(train.size_cluster, columns=['size_0', 'size_1', 'size_2'])
    train = pd.concat([train, dummies, dummie, dum], axis=1)
    
    # Make cluster dummies for validate
    dummies = pd.get_dummies(val.mvp_cluster, columns=['mvp_0', 'mvp_1', 'mvp_2', 'mvp_3'])
    dummie = pd.get_dummies(val.value_cluster, columns=['value_0', 'value_1', 'value_2'])
    dum = pd.get_dummies(val.size_cluster, columns=['size_0', 'size_1', 'size_2'])
    val = pd.concat([val, dummies, dummie, dum], axis=1)
    # fix the labeling since there are no homes in size_1 cluster in the validate dataset
    val['size_1'] = 0
    
    # Make cluster dummies for test
    dummies = pd.get_dummies(test.mvp_cluster, columns=['mvp_0', 'mvp_1', 'mvp_2', 'mvp_3'])
    dummie = pd.get_dummies(test.value_cluster, columns=['value_0', 'value_1', 'value_2'])
    dum = pd.get_dummies(test.size_cluster, columns=['size_0', 'size_1', 'size_2'])
    test = pd.concat([test, dummies, dummie, dum], axis=1)
    
    # rename the columns so the dummies are all human readable
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
    # reorder the validate columns to match the order of train and test
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
    

def get_q1_vis(train):
    '''
    This function takes in the train dataframe and displays the visualization
    for question 1 in the zillow clustering project final report.
    '''
    # set up the correlation results
    train_corr = train[['basement_sqft', 'baths', 'beds', 'decktype', 'area', 'county', 'lat',
           'long', 'lotsize', 'rooms', 'year_built', 'structure_value', 'tax_value', 'land_value', 'taxes',
           'tax_delq_year', 'census', 'logerror', 'living_space', 'price_sqft']].corr().stack().reset_index(name="correlation")
    # create the plot
    g = sns.relplot(
        data=train_corr,
        x="level_0", y="level_1", hue="correlation", size="correlation",
        palette="icefire", hue_norm=(-1, 1), edgecolor=".7",
        height=10, sizes=(50, 250), size_norm=(-.2, .8))
    # label the visualization
    g.set(xlabel="", ylabel="", title='Zillow Correlation Scatterplot heatmap', aspect="equal")
    # format it
    g.despine(left=True, bottom=True)
    g.ax.margins(.02)
    # highlight in red the target variable row
    g.map(plt.axhline, y=17, color='red', zorder=1,linewidth=20, alpha=.2)
    # format the labels!
    for label in g.ax.get_xticklabels():
        label.set_rotation(90)
    for artist in g.legend.legendHandles:
        artist.set_edgecolor(".7")

    plt.show()
    
    
def get_q2_vis(train):
    '''
    This function takes in the train dataframe and displays the visualization
    for question 2 in the zillow clustering project final report.
    '''
    # make it big
    plt.figure(figsize=(12,8))
    # plot it
    sns.histplot(data=train, x='logerror')
    # make a box to show where the next plot 'zooms' into
    plt.hlines(0,-2,2, color='red')
    plt.hlines(50,-2,2, color='red')
    plt.vlines(-2,0,50, color='red')
    plt.vlines(2,0,50, color='red')
    plt.xlim(-3,3)
    plt.ylim(0,1100)
    # give it a title
    plt.title('It is difficult to visualize the outliars in logerror without zooming in')
    plt.show()
    
    # make it big
    plt.figure(figsize=(12,8))
    # plot it
    sns.histplot(data=train, x='logerror')
    plt.axvline((train.logerror.mean() + train.logerror.std()), color='black')
    plt.annotate(text='', xytext=((train.logerror.mean() + train.logerror.std()),35), xy=(.5,35), arrowprops={'facecolor':'black'})
    plt.text(x=.55,y=34.75, s='Above 1 Standard Deviation')
    plt.axvline((train.logerror.mean() - train.logerror.std()), color='black')
    plt.annotate(text='', xytext=((train.logerror.mean() - train.logerror.std()),35), xy=(-.5,35), arrowprops={'facecolor':'black'})
    plt.text(x=-1.4,y=34.75, s='Below 1 Standard Deviation')
    plt.xlim(-2,2)
    plt.ylim(0,50)
    
    # give it a title
    plt.title('''Logerror's distribution is normal''')
    plt.show()
    
    
def get_q2_stats(train):
    '''
    This function takes in the train dataframe and returns the statistical results 
    for the hypothesis paired with question 2 in the zillow clustering project 
    final report.
    '''
    # create the samples
    below = train[train.logerror < train.logerror.mean()-train.logerror.std()].logerror
    above = train[train.logerror > train.logerror.mean()+train.logerror.std()].logerror
    # set alpha
    α = 0.05
    # run the levene test
    t, pval = stats.levene(below, above)
    # run the ttest with equal_var False
    t, p = stats.ttest_ind(below, above, equal_var=False)
    
    # Determine the results
    if p < α and t > 0:
        print('''Reject the Null Hypothesis.
Findings suggest homes with positive logerror have a higher mean absolute value logerror than homes with a negative logerror.        ''')
    else:
        print('''Fail to Reject the Null Hypothesis.
Findings suggest homes with positive logerror have a lower or equal mean absolute value logerror than homes with a negative logerror.''')
        
        
        
def get_q3_vis(train):
    '''
    This function takes in the train dataframe and displays the visualization
    for question 3 in the zillow clustering project final report.
    '''
    # create the plot
    g = sns.relplot(data=train[(train.logerror > train.logerror.mean()+train.logerror.std())|(train.logerror < train.logerror.mean()-train.logerror.std())], 
            x='lat', 
            y='long', 
            hue='logerror', 
            palette='afmhot',
            col='mvp_cluster', 
            col_wrap=2)
    # adjust the top to fit the title
    g.fig.subplots_adjust(top=0.9)
    # give it a title
    g.fig.suptitle('Cluster 3 holds few properties with logerror outside of one standard deviation')
    plt.show()
    
def get_q3_stats(train):
    '''
    This function takes in the train dataframe and returns the statistical results 
    for the hypothesis paired with question 3 in the zillow clustering project 
    final report.
    '''
    # set the population mean
    μ = train.tax_value.mean()
    # create the sample
    c3 = train[train.mvp_cluster == 3].tax_value
    # set alpha
    α = 0.05
    # to the test
    t, p = stats.ttest_1samp(c3, μ)
    # Determine the results
    if p/2 < α and t < 0:
        print('''Reject the Null Hypothesis.
    Findings suggest cluster 3 has a lower mean tax value than the population.''')
    else:
        print('''Fail to Reject the Null Hypothesis.
    Findings suggest cluster 3 has a greater than or equal mean tax value to the population.''')
        
        
def get_q4_vis(train):
    '''
    This function takes in the train dataframe and displays the visualization
    for question 4 in the zillow clustering project final report.
    '''
    # plot it
    g = sns.relplot(data=train[(train.logerror > train.logerror.mean()+train.logerror.std())|(train.logerror < train.logerror.mean()-train.logerror.std())], 
            x='lat', 
            y='long', 
            hue='logerror', 
            palette='afmhot',
            col='value_cluster')
    # format it to fit the title
    g.fig.subplots_adjust(top=0.9)
    # give it a title
    g.fig.suptitle('Cluster 0 includes most homes with logerror outside of one standard deviation')
    plt.show()
    

def get_q4_stats(train):
    '''
    This function takes in the train dataframe and returns the statistical results 
    for the hypothesis paired with question 4 in the zillow clustering project 
    final report.
    '''
    # create the sample
    c0 = train[train.value_cluster==0].tax_value
    # create the population mean
    μ = train.tax_value.mean()
    # set alpha
    α = 0.05
    # do the test
    t, p = stats.ttest_1samp(c0, μ)
    # determine the results
    if p/2 < α and t < 0:
        print('''Reject the Null Hypothesis.
    Findings suggest the mean tax value of cluster 0 is less than the population.''')
    else:
        print('''Fail to Reject the Null Hypothesis.
    Findings suggest the mean tax value of cluster 0 is greater than or equal to the population.''')
        
        
def get_q5_vis(train):
    '''
    This function takes in the train dataframe and displays the visualization
    for question 5 in the zillow clustering project final report.
    '''
    # plot it
    g = sns.relplot(data=train[(train.logerror > train.logerror.mean()+train.logerror.std())|(train.logerror < train.logerror.mean()-train.logerror.std())], 
            x='lat', 
            y='long', 
            hue='logerror', 
            palette='afmhot',
            col='size_cluster')
    # format it to fit the title
    g.fig.subplots_adjust(top=0.9)
    # give it a title
    g.fig.suptitle('Cluster 0 includes almost all homes with logerror outside of one standard deviation')
    plt.show()
    
    
def get_q5_stats(train):
    '''
    This function takes in the train dataframe and returns the statistical results 
    for the hypothesis paired with question 5 in the zillow clustering project 
    final report.
    '''
    # create the sample
    c0 = train[train.size_cluster== 0].living_space
    # create the population mean
    μ = train.living_space.mean()
    # set alpha
    α = 0.05
    # run the test
    t, p = stats.ttest_1samp(c0, μ)
    # determine the results
    if p/2 < α and t > 0:
        print('''Reject the Null Hypothesis.
    Findings suggest Cluster 0 has a greater mean living space than the population.''')
    else:
        print('''Fail to Reject the Null Hypothesis.
    Findings suggest Cluster 0 has a less than or equal mean living space than the population.''')
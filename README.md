# <a name="top"></a>Zillow Clustering Project
![]()

by: Michael Haerle & Morgan Cross

<p>
  <a href="https://github.com/michael-haerle" target="_blank">
    <img alt="Michael" src="https://img.shields.io/github/followers/michael-haerle?label=Follow_Michael&style=social" />
  </a>
</p>
<p>
  <a href="https://github.com/morgancross2" target="_blank">
    <img alt="Morgan" src="https://img.shields.io/github/followers/morgancross2?label=Follow_Morgan&style=social" />
  </a>
</p>

***
[[Project Description](#project_description)]
[[Project Planning](#planning)]
[[Key Findings](#findings)]
[[Data Dictionary](#dictionary)]
[[Data Acquire and Prep](#wrangle)]
[[Data Exploration](#explore)]
[[Statistical Analysis](#stats)]
[[Modeling](#model)]
[[Conclusion](#conclusion)]
___


## <a name="project_description"></a>Project Description:
Using the data science pipeline to practice with regression using clustering. In this repository you will find everything you need to replicate this project. This project uses the Zillow dataset to find key drivers of property value. 

[[Back to top](#top)]

***
## <a name="planning"></a>Project Planning: 
[[Back to top](#top)]

### Project Outline:
- Create README.md with data dictionary, project and business goals, come up with questions to lead the exploration and the steps to reproduce.
- Acquire data from the Codeup Database and create a function to automate this process. Save the function in an wrangle.py file to import into the Final Report Notebook.
- Clean and prepare data for exploration. Create a function to automate the process, store the function in the wrangle.py module, and prepare data in Final Report Notebook by importing and using the funtion.
- Produce at least 7 clean and easy to understand visuals.
- Clearly define hypotheses, set an alpha, run the statistical tests needed, reject or fail to reject the Null Hypothesis, and document findings and takeaways.
- Scale the data for modeling.
- Establish a baseline accuracy.
- Train three different classification models.
- Evaluate models on train and validate datasets.
- Choose the model with that performs the best and evaluate that single model on the test dataset.
- Document conclusions, takeaways, and next steps in the Final Report Notebook.


### Project goals: 
- Identify factors evaluated in home value
- Build a model to best predict home value
- Minimize Root Square Mean Error (RMSE) in modeling


### Target variable:
- The target variable for this project is logerror.

### Initial questions:
- What continuous features have a relationship with logerror?
- Where is the most logerror?
- If we cluster our basic home features, is there a relationship with logerror?
- If we cluster our value related features, is there a relationship with logerror?
- If we cluster our size related features, is there a relationship with logerror?

### Need to haves (Deliverables):
- A final report notebook
- A 5min recorded presentation


### Nice to haves (With more time):
 - Develop a model using different machine learning techniques focused on the outliers. The more we can learn about what makes them outliers, will lead us to what is causing the error.


### Steps to Reproduce:
- You will need to make an env.py file with a vaild username, hostname and password assigned to the variables user, host, and password
- Then download the wrangle.py, explore.py, model.py, and final_report.ipynb
- Make sure these are all in the same directory and run the final_report.ipynb.

***

## <a name="findings"></a>Key Findings:
[[Back to top](#top)]

- Without cluster, there is no obvious continuous feature that has a relationship with logerror.
- Clustering with tax_value seems to play a key role in logerror.
- Clusering by size does not seem to create useful splits in the data from what is human readable. More time is needed to explore this finding.


***

## <a name="dictionary"></a>Data Dictionary  
[[Back to top](#top)]

### Data Used
---
| Target | Type | Description |
| ---- | ---- | ---- |
| logerror | float | The log of the error in the zestimate model |


| Feature Name | Type | Description |
| ---- | ---- | ---- |
| area | float | Sum of square feet in the home |
| area12 | float | Finished living area |
| assessment_year| float | year the home was assessed |
| basement_sqft | float |  Finished living area below or partially below ground level |
| bathnbed | float | Number of bathrooms and bedrooms combined |
| baths | float | Count of bathrooms in the home |
| beds | float | Count of bedrooms in the home |
| census | float | tract and block data from the census |
| city_id | float | City in which the property is located (if any) |
| construction_type | str | type of construction the home is classified |
| county | float | Fips code for the county the home is located in |
| county_id | float |County in which the property is located |
| decktype | float | Type of deck (if any) present on parcel |
| fireplace | float | Number of fireplaces in a home (if any) |
| fireplace_flag | float | 1 = Fireplace present, 0 = No Fireplace |
| fullbath | float | Number of full bathrooms (sink, shower + bathtub, and toilet) present in home |
| hottub_or_spa | float | Does the home have a hot tub or spa |
| land_value | float | tax assessed value of the land only |
| landuse_code | str | County land use code i.e. it's zoning at the county level |
| landuse_desc | str | description of the land use |
| lat | float | The home's geographical latitude |
| living_space | float | The home area in sqft minus 200sqft per bedroom and 60sqft per bathroom (average sqft per respective room) |
| long | float | The home's geographical longitude |
| lot_size | float | Sum of square feet of the piece of land the home is on |
| mvp_cluster | int | clusters built using primary home features |
| mvp_0 | unit | cluster 0 of mvp clusters |
| mvp_1 | unit | cluster 1 of mvp clusters |
| mvp_2 | unit | cluster 2 of mvp clusters |
| mvp_3 | unit | cluster 3 of mvp clusters |
| pool | float | Number of pools on the lot (if any) |
| pool10 | float | Spa or Hot Tub |
| pool2 | float | Pool with Spa/Hot Tub |
| pool7 | float | Pool without hot tub |
| price_sqft | float | the home tax_value over the home's area in sqft |
| raw_census | float | Census tract and block ID combined - also contains blockgroup assignment by extension |
| rooms | float | Total number of rooms in the principal residence |
| size_cluster | int | clusters built using size features |
| size_0 | unit | cluster 0 of size clusters |
| size_1 | unit | cluster 1 of size clusters |
| size_2 | unit | cluster 2 of size clusters |
| structure_value | float |The assessed value of the built structure on the parcel |
| taxes | float | taxes on the home's value |
| tax_delq_flag | str | Y if the home is deliquent on paying taxes, N if not |
| tax_delq_year | float | year the home became deliquent on paying taxes, 9999 if not |
| tax_value | float | tax assessed value of the home |
| threequarterbnb | float |Count of three-quarter bathrooms (if any) |
| transactiondate | str | date the home transaction took place |
| value_cluster | int | clusters built using assessed value features |
| value_0 | unit | cluster 0 of value clusters |
| value_1 | unit | cluster 1 of value clusters |
| value_2 | unit | cluster 2 of value clusters |
| yard_size | float | The lot size minus the home area in sqft |
| year_built | float | The year the home was built |
| zip_id | float | Zip code in which the property is located |
***

## <a name="wrangle"></a>Data Acquisition and Preparation
[[Back to top](#top)]

![]()


### Prepare steps: 
- Droped duplicate columns
- Created dummy variables
- Concatenated the dummy dataframe
- Renamed columns
- Dropped columns not needed
- Removed ouliers
- Imputed null values
- Split into the train, validate, and test sets

*********************

## <a name="explore"></a>Data Exploration:
[[Back to top](#top)]
- Python files used for exploration:
    - wrangle.py
    - explore.py
    - model.py


### Takeaways from exploration:
- Without cluster, there is no obvious continuous feature that has a relationship with logerror.
- Clustering with tax_value seems to play a key role in logerror.
- Clusering by size does not seem to create useful splits in the data from what is human readable. More time is needed to explore this finding.

***

## <a name="stats"></a>Statistical Analysis
[[Back to top](#top)]

### Stats Test 1: T-Test: One-Sample, One-Tail


#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is: The absolute value mean logerror of homes with positive logerror is less than or equal to the absolute value mean logerror of homes with negative logerror.
- The alternate hypothesis (H<sub>1</sub>) is: The absolute value mean logerror of homes with positive logerror is greater than the absolute value mean logerror of homes with negative logerror.

#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05

#### Results:
- Fail to Reject the Null Hypothesis.
- Findings suggest homes with positive logerror have a lower or equal mean absolute value logerror than homes with a negative logerror.


### Stats Test 2: T-Test: One-Sample, One-Tail


#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is: Cluster 3 has a greater than or equal mean tax value than the population.
- The alternate hypothesis (H<sub>1</sub>) is: Cluster 3 has a lower mean tax value than the population.

#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05


#### Results:
- Reject the Null Hypothesis.
- Findings suggest cluster 3 has a lower mean tax value than the population.


### Stats Test 3: T-Test: One-Sample, One-Tail


#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is: Cluster 0 has a greater than or equal mean tax value than the population.
- The alternate hypothesis (H<sub>1</sub>) is: Cluster 0 has a lower mean tax value than the population.

#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05


#### Results:
- Reject the Null Hypothesis.
- Findings suggest the mean tax value of cluster 0 is less than the population.


### Stats Test 4: T-Test: One-Sample, One-Tail


#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is: Cluster 0 has a less than or equal mean living space than the population.
- The alternate hypothesis (H<sub>1</sub>) is: Cluster 0 has a greater mean living space than the population.

#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05


#### Results:
- Fail to Reject the Null Hypothesis.
- Findings suggest Cluster 0 has a less than or equal mean living space than the population.

***

## <a name="model"></a>Modeling:
[[Back to top](#top)]

### Baseline (Using Mean)
    
- Baseline RMSE: 0.177473
    

- Selected features to input into models:
    - features =  ['baths', 'beds', 'living_space', 'county', 'lat', 'long', 'lotsize', 'year_built', 'tax_value', 'price_sqft', 'mvp_0', 'mvp_2', 'value_2']

***

## Models:


### Model 1: OLS using LinearRegression


Model 1 results:
- RMSE for OLS using LinearRegression
- Training/In-Sample:  0.177096
- Validation/Out-of-Sample:  0.160083
- R2 Value: 0.006709


### Model 2 : Lars + Lasso, Alpha 1


Model 2 results:
- RMSE for OLS using Lars + Lasso
- Training/In-Sample:  0.177473	
- Validation/Out-of-Sample:  0.160623
- R2 Value: 0.000000


### Model 3 : Quadratic Linear Regression

Model 3 results:
- RMSE for Quadratic Linear Regression
- Training/In-Sample:  0.176301 
- Validation/Out-of-Sample:  0.159838
- R2 Value: 0.009738


### Model 4 : Cubic Linear Regression

Model 4 results:
- RMSE for Cubic Linear Regression
- Training/In-Sample:  0.172618
- Validation/Out-of-Sample:  0.271843
- R2 Value: -1.863983


## Selecting the Best Model:

### Use Table below as a template for all Modeling results for easy comparison:

| Model | Validation | R2 |
| ---- | ---- | ---- |
| Baseline | 0.160623 | 0.000000 |
| OLS using LinearRegression | 0.160083 | 0.006709 |
| Lars + Lasso | 0.160623 |  0.000000 |
| Quadratic Linear Regression | 0.159838 | 0.009738 |
| Cubic Linear Regression | 0.271843 | -1.863983 |


- {OLS using LinearRegression} model performed the best


## Testing the Model

- Model Testing Results: RMSE 0.182309, R2 0.002744

***

## <a name="conclusion"></a>Conclusion:

- Pinpointing the source of the logerror in the zestimate model has proven to be complex. Exploring with clustering algorithms can better this model, but negligibly. 
- The best model only reduced the root mean squared error by 0.3% from the baseline results.

#### Implement policies requiring complete information input when adding or update home information. This will reduce the missing values in the dataset and could reduce error.

[[Back to top](#top)]

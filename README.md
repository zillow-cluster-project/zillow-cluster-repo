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
- The target variable for this project is log error.

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
 - If we had more time we would like to explore the size cluster we created more.
 - We would also like to try more combinations for clustering and modeling.


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
- Imputed nulls with 0 for fireplace and full_bathroom
- Used square feet to feature engineer a new column where it returned small, medium, or large house size
- Used .apply to apply a custom function to create a decade column for what decade the house was built in
- Converted latitude and longitude to the proper values
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

### Stats Test 1: Chi Square


#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is: The Decade Built and Tax Value are independent.
- The alternate hypothesis (H<sub>1</sub>) is: There is a relationship between tax value and the Decade Built.

#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05

#### Results:
- We reject the null hypothesis that The Decade Built and Tax Value are independent
- There is a relationship between tax value and the Decade Built
- 3.9309219442730487e-16
- Chi2 214095.42
- Degrees of Freedom 208846


### Stats Test 2: Chi Square


#### Hypothesis:
- The null hypothesis (H<sub>0</sub>) is: The Year Built in LA and Tax value are independent.
- The alternate hypothesis (H<sub>1</sub>) is: There is a relationship between Tax Value and the Year Built in LA.

#### Confidence level and alpha value:
- I established a 95% confidence level
- alpha = 1 - confidence, therefore alpha is 0.05


#### Results:
- We fail to reject the null hypothesis that The Year Built in LA and Tax value are independent
- There appears to be no relationship between Tax Value and the Year Built in LA
- P-Value 0.4721751128088897
- Chi2 1454990.41
- Degrees of Freedom 1454872


***

## <a name="model"></a>Modeling:
[[Back to top](#top)]

### Baseline (Using Mean)
    
- Baseline RMSE: 247730.36
    

- Selected features to input into models:
    - features =  ['bedrooms', 'bathrooms', 'square_feet', 'lot_square_feet', 'full_bathroom', 'year_built', 'fips', 'region_zip', 'house_size_large', 'house_size_small', 'decade']

***

## Models:


### Model 1: Lasso + Lars


Model 1 results:
- RMSE for Lasso + Lars
- Training/In-Sample:  212401.75 
- Validation/Out-of-Sample:  216116.17
- R2 Value: 0.26


### Model 2 : OLS using LinearRegression


Model 2 results:
- RMSE for OLS using LinearRegression
- Training/In-Sample:  212395.09 
- Validation/Out-of-Sample:  216108.91
- R2 Value: 0.26


### Model 3 : Polynomial Model

Model 3 results:
- RMSE for Polynomial Model, degrees=2
- Training/In-Sample:  204839.27 
- Validation/Out-of-Sample:  208981.93
- R2 Value: 0.31


## Selecting the Best Model:

### Use Table below as a template for all Modeling results for easy comparison:

| Model | Validation | R2 |
| ---- | ---- | ---- |
| Baseline | 247730.36 | 0.0 |
| Lasso + Lars | 216116.17 | 0.26 |
| OLS using LinearRegression | 216108.91 |  0.26 |
| Polynomial Model | 208981.93 | 0.31 |


- {Polynomial Model} model performed the best


## Testing the Model

- Model Testing Results: RMSE 204854.96, R2 0.31

***

## <a name="conclusion"></a>Conclusion:

- Tax Value has a positive correlation with house_size_large, decade, full_bathroom, year_built, square_feet, bathrooms, and bedrooms.
- Any decade after the 1960's is above the average Tax Value.
- Our RMSE value for our test dataset beat our baseline by 41,831.16.

#### A way to further improve the our predictions would be ensuring that the data gathered didn't have as many nulls, and a catagory to select if its a certain distance away from a beach. There was an extreme amount of nulls in the data, this is definitely the best way to improve predictions.

[[Back to top](#top)]

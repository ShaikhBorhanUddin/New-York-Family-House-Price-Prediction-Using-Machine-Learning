# New York Family House Price Prediction Using Machine Learning 

<p align="left">
  <!-- Core -->
  <img src="https://img.shields.io/badge/Made%20With-Colab-blue?logo=googlecolab&logoColor=white" alt="Made with Colab">
  <img src="https://img.shields.io/badge/Language-Python-green?logo=python" alt="Language: Python">
  <img src="https://img.shields.io/badge/💻Dev%20Environment-VS%20Code-blue?logo=visualstudiocode">

  <!-- License & Issues -->
  <img src="https://img.shields.io/badge/⚖️%20License-MIT-red" alt="License">
  <img src="https://img.shields.io/badge/🐞%20Issues-None-green" alt="Issues">

  <!-- Repo Stats -->
  <img src="https://img.shields.io/github/repo-size/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning?logo=github" />
  <img src="https://img.shields.io/github/last-commit/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning" alt="Last Commit">

  <!-- Models -->
  <img src="https://img.shields.io/badge/🤖%20Models-XGBoost | ElasticNet | Random Forest-red" alt="Model: XGBoost">

  <!-- Dataset -->
  <img src="https://img.shields.io/badge/🗂️Dataset-Kaggle | data.gov | datahub-blueviolet" alt="Dataset: NYC Property Sales"> 

  <!-- Runtime -->
  <img src="https://img.shields.io/badge/⚙️Runtime-CPU-blue" alt="Runtime"> 
  
  <!-- Visualization -->
  <img src="https://img.shields.io/badge/📊%20Visualization-Matplotlib | Seaborn | Folium-yellow" alt="Visualization">

  <!-- Deployment -->
  <img src="https://img.shields.io/badge/Deployment-Streamlit-orange?logo=streamlit" alt="Deployment: Streamlit">

  <!-- DevOps -->
  <img src="https://img.shields.io/badge/Version%20Control-Git-orange?logo=git" alt="Git">
  <img src="https://img.shields.io/badge/Host-GitHub-green?logo=github" alt="GitHub">

  <!-- Social -->
  <img src="https://img.shields.io/github/forks/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning?style=social" alt="Forks">

  <!-- Status -->
  <img src="https://img.shields.io/badge/🏁Project-Deployed-brightgreen" alt="Status">
</p> 

![Dashboard](https://github.com/ShaikhBorhanUddin/New-York-Property-Price-Prediction-Using-Machine-Learning/blob/main/Assets/nyc_title.png?raw=true) 

This project focuses on predicting sale prices for low-density residential family houses in New York City, specifically one-, two-, and three-family dwellings and homes. Multi-unit apartment buildings, condominiums, cooperatives, rentals, and commercial properties were excluded to ensure homogeneous pricing behavior and reliable model performance. 

## Project Objective 

Accurately estimating residential property prices in New York City is challenging due to market volatility, location-based variation, and heterogeneous housing characteristics. This project aims to build a machine learning model that predicts family home prices using historical NYC property sales data and deploys the model as an interactive web application. 

## Folder Structure 

```bash
House Price Prediction Project
│
├── Assets/                                  # Images for project documentation
├── Dataset/               
│      ├── Raw/                  
│      │     ├── nyc_property_sales.csv      # Not included in repository due to large size (364.5 MB)
│      │     ├── PLUTO.csv                   # Not included in repository due to large size (292.1 MB)
│      │     └── cpi_index.csv 
│      ├── Cleaned/               
│      │     ├── nyc_property_sales_cleaned.csv
│      │     ├── PLUTO_cleaned.csv
│      │     └── cpi_index_clean.csv
│      └── Feature Engineered/
│            └── sales_pluto_cpi_combined_engineered.csv
├── Models/
│      ├── combined_location_mapping.pkl                       
│      ├── feature_names.pkl
│      ├── location_coordinate_mapping.pkl
│      ├── unique_categorical_values.pkl
│      ├── xgboost_model.pkl
│      ├── elastic_net_model.pkl
│      └── rf_model.pkl                       # Not included in repository due to large size (3.66 GB)
├── Notebooks/                                
│      ├── property_sales_data_preprocessing.ipynb
│      ├── PLUTO_data_preprocessing.ipynb
│      ├── CPI_data_preprocessing.ipynb
│      ├── EDA.ipynb
│      ├── feature_engineering.ipynb
│      ├── xgboost_train_test.ipynb
│      ├── elasticnet_train_test.ipynb
│      ├── rf_train_test.ipynb
│      └── model_comparison.ipynb
├── app.py                                    # Deployment code
├── requirements.txt                          # Python dependencies for deployment
├── README.md                                 # Project documentation
└── Licence
```

## Workflow 

## Dataset 

The primary dataset [Current NYC Property Sales](https://www.kaggle.com/datasets/datasciencedonut/current-nyc-property-sales), is sourced from Kaggle. The dataset contains a record of every property sold in the New York City property market since 2003 (the first year sales data was first listed on the public record) and updates monthly to include rolling sales. However, for this project, 2018049 records from 2003 to 2023 were considered. This dataset is mainly used for property descriptions and sales transaction information during machine learning model training. A brief description of columns of the Sales dataset is given in the following table. 

| Columns | Data Type | Description |
|-----------|-----------|-------------|
| BOROUGH | object | The name of the borough in which the property is located. |
| NEIGHBORHOOD | object | The common name of the neighborhood. |
| BUILDING CLASS CATEGORY | object | Classification to easily identify similar properties by broad usage. |
| TAX CLASS AT PRESENT | object | Tax class assigned to the property based on its current use. |
| BLOCK | object | A tax block is a subdivision of the borough on which real properties are located. |
| LOT | object | A tax lot is a subdivision of a tax block that uniquely identifies a property. |
| EASE-MENT | object | A legal right allowing limited use of another’s property (e.g., right of way). |
| BUILDING CLASS AT PRESENT | object | Classification describing the property's current constructive use. |
| ADDRESS | object | The street address of the property. |
| APARTMENT NUMBER | object | Apartment number of the property, if applicable. |
| ZIP CODE | object | The ZIP code of the property. |
| RESIDENTIAL UNITS | object | Number of residential units contained in the property. |
| COMMERCIAL UNITS | object | Number of commercial units contained in the property. |
| TOTAL UNITS | object | Total number of units in the property. |
| LAND SQUARE FEET | object | Land area of the property in square feet. |
| GROSS SQUARE FEET | object | Total floor area of the building measured from exterior walls. |
| YEAR BUILT | object | Year the structure on the property was built. |
| TAX CLASS AT TIME OF SALE | object | Tax class assigned to the property at the time of sale. |
| BUILDING CLASS AT TIME OF SALE | object | Building classification at the time of sale. |
| SALE PRICE | object | The price for which the property was sold. |
| SALE DATE | object | The date when the property was sold. | 

However, the primary dataset does not include precise geolocation information (latitude and longitude), which is crucial for accurately pinpointing properties on a map during deployment. Therefore, to enrich the property location details, a secondary [PLUTO dataset](https://catalog.data.gov/dataset/primary-land-use-tax-lot-output-pluto) is used. This dataset (858284 entries) contains detailed property attributes beyond those available in the sales dataset, such as land area and building area, including residential area, office area, garage area, commercial area, frontage, land depth, storage area, and more. While these features provide rich spatial and structural information, their high dimensionality may introduce noise and potentially overwhelm machine learning models if not carefully selected or engineered. Summary of important columns of this dataset is given below. 

| Columns | Data Type | Description |
|------------|-----------|-------------|
| borough | object | The name of the borough where the property is located. |
| Tax block | int64 | A tax block is a subdivision of the borough used to identify real property locations. |
| Tax lot | int64 | A tax lot is a subdivision of a tax block that uniquely identifies a property. |
| postcode | float64 | The ZIP/postal code of the property location. |
| yearbuilt | float64 | The year in which the property structure was originally built. |
| latitude | float64 | The geographic latitude coordinate of the property location. |
| longitude | float64 | The geographic longitude coordinate of the property location. | 

The sale history in the primary dataset spans more than 20 years. To accurately predict house prices in the context of present-day economic conditions, inflation must be taken into account. Therefore, a [Consumer Price Index (CPI) dataset](https://datahub.io/core/cpi-us) is used to adjust historical sale prices to their current-time equivalents. This dataset contains the monthly CPI index from January 1913 to December 2025, though only the data corresponding to the 20-year sales period will be utilized for this project. Inflation adjustment is an industry-standard practice, widely adopted by economists, financial institutions, and real estate analytics platforms such as Zillow and Redfin, all of which rely on monthly CPI data for price normalization. Initial shape of CPI Index dataset is as follows. 

| Columns | Data Type | Description |
|-----------|-----------|-------------|
| Date | object | The date corresponding to the inflation record. |
| Index | object | The inflation index value used for price normalization. |
| Inflation | object | The inflation rate representing the change in price level over time. | 

The property sales dataset contains a significant number of missing values in several features, such as building unit type, size, apartment number, and easement. In contrast, the PLUTO dataset has far fewer missing values, with fewer than 2,000 records missing in fields such as postcode, latitude, and longitude. 

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/sales_null_value_count.png" width="46.45%" />
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/PLUTO_null_value_count.png" width="53%" />
</p> 

Handling these missing values, along with data cleaning, feature modification, extraction of new features, and the subsequent merging of the three datasets, will be discussed in detail in the following sections. 

## Data Cleaning and Preprocessing 

Data cleaning begins with the property sales dataset. The **BOROUGH** column contains mixed data types (integers and strings). Converting all values to integers reduces the column to five unique entries, corresponding to the five boroughs of New York City. 

```bash
1: 'Manhattan', 2: 'Bronx', 3: 'Brooklyn', 4: 'Queens', 5: 'Staten Island'
```
In the **NEIGHBORHOOD** column, inconsistencies were observed due to trailing white spaces, which artificially inflated the number of unique values (for example, both 'ALPHABET CITY' and 'ALPHABET CITY&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;' appeared as separate entries). Trimming white spaces, along with removing anomalous non-text entries (such as integer values like 1026, 3019, 3004, and 1021), significantly reduced redundant categories and improved data consistency. 

Each **BUILDING CLASS CATEGORY** entry combines a category number and a textual description, where the first two characters represent the category code and the remaining text provides the description. To facilitate more effective machine learning training, this feature was decomposed into two separate variables: **BUILDING CLASS CATEGORY NUMBER** and **BUILDING CLASS CATEGORY DESCRIPTION**. 

|     BUILDING CLASS CATEGORY    | BUILDING CLASS CATEGORY NUMBER | BUILDING CLASS CATEGORY DESCRIPTION |
|--------------------------------|--------------------------------|-------------------------------------|
|     01 ONE FAMILY DWELLINGS    |               1                |        ONE FAMILY DWELLINGS         |
|     02 TWO FAMILY DWELLINGS    |               2                |        TWO FAMILY DWELLINGS         |
| 07 RENTALS - WALKUP APARTMENTS |               7                |      RENTALS - WALKUP APARTMENTS    | 

Since this project focuses on family house price prediction, only records corresponding to one-, two-, and three-family homes/dwellings were retained, while all other property categories were excluded. Although this filtering significantly reduced the dataset size, it preserved a sufficiently large and representative sample for modeling family housing prices.

The **ZIP CODE** column name contained embedded whitespace, which was stripped to standardize the schema and prevent downstream processing issues. The **APARTMENT NUMBER** feature was removed, as apartment identifiers are not predictive of sale prices and may introduce noise (different properties can share identical apartment numbers, which can confuse machine learning models).

The **ADDRESS** column was also dropped due to its high risk of data leakage and extremely high cardinality, with over 600,000 unique textual values, making it impractical for encoding prior to model training. Similarly, the **EASE-MENT** column contained a substantial proportion of missing values and was therefore excluded from further analysis. Since the **BUILDING CLASS CATEGORY NUMBER** and **BUILDING CLASS CATEGORY DESCRIPTION** features were derived from the original **BUILDING CLASS CATEGORY** field, the original column became redundant and was subsequently removed.

While the original **SALE DATE** provides day-level granularity, real estate prices typically vary at monthly or quarterly scales. To capture meaningful temporal trends while avoiding high-cardinality features and potential leakage, **SALE YEAR** and **SALE MONTH** were derived from the sale date and retained in place of the full date field. 

| SALE DATE | SALE YEAR | SALE MONTH |
|-----------|-----------|------------|
| 2022-09-29 | 2022 | 9 |
| 2022-07-28 | 2022 | 7 |
| 2022-04-08 | 2022 | 8 | 

Finally, all **SALE PRICE** entries below $15,000 were removed from the dataset. This threshold is not arbitrary but reflects an industry-informed heuristic based on New York City real estate behavior and dataset characteristics. In the NYC market, even the lowest-priced legitimate transactions (such as vacant land, foreclosed units, or parking spaces) rarely occur below $15,000 in arm’s-length sales. Moreover, the dataset documentation indicates that zero-valued sales represent transfers without cash consideration, and exploratory analysis shows that most non-market or artificial transactions cluster between $0 and $10,000. Setting the cutoff at $15,000 effectively removes these non-genuine transactions while preserving valid low-value sales, resulting in a cleaner and more reliable target variable for modeling.

At this stage, the [cleaned property sales dataset](https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Dataset/Cleaned/nyc_property_sales_cleaned.csv.zip) is finalized and saved for subsequent use in the feature engineering pipeline. 

The PLUTO (Primary Land Use Tax Lot Output) dataset provides extensive land-use and geographic information at the tax lot level. To reduce complexity during the feature engineering process, only a subset of relevant features was retained, including **borough**, **Tax block**, **Tax lot**, **postcode**, **year built**, **latitude**, and **longitude**, while all other columns were excluded. 

To maintain data integrity and ensure consistency across datasets, the following mapping was applied to the borough column. 

```bash
'MN': 1, 'BX': 2, 'BK': 3, 'QN': 4, 'SI': 5
```

Once these steps were completed, the [cleaned PLUTO dataset](https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Dataset/Cleaned/PLUTO_cleaned.csv.zip) was saved for use in the subsequent feature engineering process. 

The CPI Index dataset was used exclusively to convert historical house sale prices to their present-time equivalents. Since the index values are already provided, the **Inflation** column was redundant and therefore removed, and the **Index** column was renamed to **CPI** for clarity and consistency. 

Official CPI figures are published by the U.S. Bureau of Labor Statistics on the first day of each month and remain valid for the entire month. To support feature engineering and dataset integration, **Year** and **Month** features were derived from the **Date** column. This enables seamless merging with the sales dataset, which contains corresponding sale year and sale month fields. 

| Date | Year | Month |
|------|------|-------|
| 2023-12-01 | 2023 | 12 |
| 2023-11-01 | 2023 | 11 |
| 2023-10-01 | 2023 | 10 | 

Upon completion of these steps, the [cleaned CPI Index dataset](https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Dataset/Cleaned/cpi_index_clean.csv) was saved for use in the subsequent feature engineering process. 

## Feature Engineering 

As discussed in the dataset section, the Property Sales dataset does not contain precise geolocation information, making it necessary to enrich the data before proceeding further. However, the cleaned Property Sales and PLUTO datasets use different column names for the common identifiers (borough, block, and lot). These columns were first standardized using a consistent naming convention (shown below). 

```bash
'borough': 'BOROUGH', 'Tax block': 'BLOCK', 'Tax lot': 'LOT'
```

Using these three standardized keys, the Sales dataset was then merged with the PLUTO dataset via a left join to retain all sales records while appending available geolocation attributes. The join logic is illustrated below. 

<p align="left">
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/sales_PLUTO_join.png" width="56%" />
</p>  

The **YEAR BUILT** column contained only a small number of missing values, which were successfully imputed using the corresponding **yearbuilt** column. In contrast, more than 7,000 records were missing **latitude** and **longitude** values. To address this, geolocation imputation was performed in a hierarchical manner. First, missing coordinates were imputed using ZIP CODE level centroids, calculated from available records, with fallback to global medians where necessary. This approach successfully resolved the majority of missing cases. For the remaining unmatched records, BOROUGH level centroids were applied to ensure complete spatial coverage. 

At this stage, the recently merged and cleaned Sales–PLUTO dataset is further integrated with the cleaned CPI dataset to enable inflation-adjusted sale price computation, using **SALE YEAR** and **SALE MONTH** from the sales data mapped to the corresponding **Year** and **Month** fields in the CPI dataset. 

```bash
'Year': 'SALE YEAR', 'Month': 'SALE MONTH'
```

Based on this year–month alignment, the datasets are merged using a left join to retain all property sales records while appending the appropriate CPI values, as illustrated in the accompanying join diagram. 

<p align="left">
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/sale_pluto_cpi_join.png" width="56%" />
</p> 

According to the U.S. Bureau of Labor Statistics, the Consumer Price Index (CPI) during January 2026 is approximately 325.17. Using this value as the reference baseline, historical sale prices are adjusted to their present-time equivalents using the formula: 

```bash
ADJUSTED SALE PRICE = SALE PRICE * ( 325.17 / CPI )
``` 

After this inflation adjustment, the sale prices become significantly more meaningful and comparable across time (illustrated in the accompanying table). 

| SALE PRICE | ADJUSTED SALE PRICE |
|-----------|-----------|
| 399000 | 437127 |
| 2999999 | 3286669 |
| 11100000 | 12181076 | 

At this stage, the original SALE PRICE column becomes redundant and is therefore removed, while ADJUSTED SALE PRICE is retained as the target variable for machine learning model training. With this, the [feature-engineered dataset](https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/tree/main/Dataset/Feature%20Engineered) is finalized and ready for exploratory data analysis, model development and training. 

## Exploratory Data Analysis 

Add text 

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/property_location_scatterplot.png" width="42.2%" />
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/borough_barchart.png" width="56.7%" />
</p> 

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/top_10_neighborhood.png" width="42.2%" />
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/zipcode_distribution.png" width="56.7%" />
</p> 

Analyze the findings 

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/built_year_map.png" width="41.8%" />
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/year_built_barchart.png" width="57.5%" />
</p> 

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/sale_year.png" width="49%" />
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/sale_month.png" width="49%" />
</p> 

Analyze the findings 

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/building_class_category_description.png" width="45.7%" />
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/building_class_at_time_of_sale.png" width="53.3%" />
</p> 

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/top_10_land_square_feet.png" width="49%" />
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/top_10_gross_square_feet.png" width="49%" />
</p> 

Analyze the findings 

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/adjusted_sale_price.png" width="55%" />
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/sales_price_map.png" width="44%" />
</p> 

Analyze the findings 

## Model Training 

## Model Performance Evaluation 

## Deployment 

To access the streamlit app click the [Link](https://nycfamilyhousepriceprediction.streamlit.app/).

## Limitations 

Since the model was trained only on family house prices, the dataset includes just a subset of all properties in each neighborhood. As a result, dynamic filtering of blocks and lots is not possible in the Streamlit app. Dynamic listing can only be enabled if the dataset contains all properties within a neighborhood. 

## Tools Used 

## Licence 

## Contact 

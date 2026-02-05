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
  <img src="https://img.shields.io/badge/📊%20Visualization-Matplotlib%20%7C%20Seaborn-yellow" alt="Visualization">

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

However, the primary dataset does not include precise geolocation information (latitude and longitude), which is crucial for accurately pinpointing properties on a map during deployment. Therefore, to enrich the property location details, a secondary [PLUTO](https://catalog.data.gov/dataset/primary-land-use-tax-lot-output-pluto) dataset is used. This dataset contains detailed property attributes beyond those available in the sales dataset, such as land area and building area, including residential area, office area, garage area, commercial area, frontage, land depth, storage area, and more. While these features provide rich spatial and structural information, their high dimensionality may introduce noise and potentially overwhelm machine learning models if not carefully selected or engineered. Summary of important columns of this dataset is given below. 

| Columns | Data Type | Description |
|------------|-----------|-------------|
| borough | object | The name of the borough where the property is located. |
| Tax block | int64 | A tax block is a subdivision of the borough used to identify real property locations. |
| Tax lot | int64 | A tax lot is a subdivision of a tax block that uniquely identifies a property. |
| postcode | float64 | The ZIP/postal code of the property location. |
| yearbuilt | float64 | The year in which the property structure was originally built. |
| latitude | float64 | The geographic latitude coordinate of the property location. |
| longitude | float64 | The geographic longitude coordinate of the property location. | 

The sale history in the primary dataset spans more than 20 years. To accurately predict house prices in the context of present-day economic conditions, inflation must be taken into account. Therefore, a [Consumer Price Index (CPI)](https://datahub.io/core/cpi-us) dataset is used to adjust historical sale prices to their current-time equivalents. This inflation adjustment is an industry-standard practice, widely adopted by economists, financial institutions, and real estate analytics platforms such as Zillow and Redfin, all of which rely on monthly CPI data for price normalization. 

| Columns | Data Type | Description |
|-----------|-----------|-------------|
| Date | object | The date corresponding to the inflation record. |
| Index | object | The inflation index value used for price normalization. |
| Inflation | object | The inflation rate representing the change in price level over time. | 

Data cleaning, feature modification, extraction of new features, and the subsequent merging of these three datasets will be discussed in detail in the following sections. 

## Data Cleaning and Preprocessing 

While exact sale dates contain day-level information, real estate prices vary at monthly or quarterly scales. Retaining year and month preserves meaningful temporal signal while avoiding high-cardinality features and potential leakage. 

Because adjusted sale price represents the inflation-normalized economic value of the property, it was used as the modeling target. Nominal sale price was excluded to avoid redundancy and inflation-driven noise. 

## Exploratory Data Analysis 

## Feature Engineering 

## Model Training 

## Model Performance Evaluation 

## Deployment 

To access the streamlit app click the [Link](https://nycfamilyhousepriceprediction.streamlit.app/).

## Limitations 

Since the model was trained only on family house prices, the dataset includes just a subset of all properties in each neighborhood. As a result, dynamic filtering of blocks and lots is not possible in the Streamlit app. Dynamic listing can only be enabled if the dataset contains all properties within a neighborhood. 

## Tools Used 

## Licence 

## Contact 

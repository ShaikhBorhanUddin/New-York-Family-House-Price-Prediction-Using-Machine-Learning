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

The original [Dataset](https://www.kaggle.com/datasets/datasciencedonut/current-nyc-property-sales) is sourced from Kaggle. For additional information on property addresses a secondary [PLUTO](https://catalog.data.gov/dataset/primary-land-use-tax-lot-output-pluto) dataset is used. For consumer price index [CPI](https://datahub.io/core/cpi-us) dataset is used. 

| Field Name | Description |
|-----------|-------------|
| BOROUGH | The name of the borough in which the property is located. |
| NEIGHBORHOOD | The common name of the neighborhood. |
| BUILDING CLASS CATEGORY | Classification to easily identify similar properties by broad usage. |
| TAX CLASS AT PRESENT | Every property in the city is assigned to one of four tax classes (Classes 1, 2, 3, and 4), based on the use of the property. |
| BLOCK | A Tax Block is a subdivision of the borough on which real properties are located. |
| LOT | A Tax Lot is a subdivision of a Tax Block and represents the property's unique location. |
| EASE-MENT | An easement is a right, such as a right of way, which allows an entity to make limited use of another’s real property. |
| BUILDING CLASS AT PRESENT | The Building Classification is used to describe a property’s constructive use. |
| ADDRESS | The street address of the property. |
| APARTMENT NUMBER | Apartment number of the property (if the property is an apartment). |
| ZIP CODE | The ZIP code of the property. |
| RESIDENTIAL UNITS | Number of residential units the property contains. |
| COMMERCIAL UNITS | Number of commercial units the property contains. |
| TOTAL UNITS | The total number of units the property contains. |
| LAND SQUARE FEET | The land area of the property listed in square feet. |
| GROSS SQUARE FEET | The total area of all the floors of a building measured from the exterior surfaces of the outside walls, including land area and internal space. |
| YEAR BUILT | Year the structure on the property was built. |
| TAX CLASS AT TIME OF SALE | The tax class assigned at the time of sale. |
| BUILDING CLASS AT TIME OF SALE | The building class assigned at the time of sale. |
| SALE PRICE | The price the property was sold for. |
| SALE DATE | The date the property was sold. | 

| Column Name | Data Type | Description |
|------------|-----------|-------------|
| borough | object | The name of the borough where the property is located. |
| Tax block | int64 | A tax block is a subdivision of the borough used to identify real property locations. |
| Tax lot | int64 | A tax lot is a subdivision of a tax block that uniquely identifies a property. |
| postcode | float64 | The ZIP/postal code of the property location. |
| yearbuilt | float64 | The year in which the property structure was originally built. |
| latitude | float64 | The geographic latitude coordinate of the property location. |
| longitude | float64 | The geographic longitude coordinate of the property location. |


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

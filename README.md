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
├── Assets/                       # Images for project documentation
├── Dataset/               
│      ├── Raw/                   # Original dataset from Kaggle (3rd dataset too large to upload in GitHub)
│      ├── Preprocessed/          # Dataset with added features
│      ├── Cleaned/               # Cleaned and merged datasets
│      └── Feature Engineered/    # dataset with BoW, TF-IDF, word2vec and lemmatized features
├── Models/                       # All saved models (distilbert safetensor too large to upload in GitHub)
├── Notebooks/                    # Data preprocessing, EDA, train/test, result visualization
├── app.py                        # Code for deployment
├── requirements.txt              # Python dependencies for deployment
├── README.md                     # Project documentation
└── Licence
```

## Workflow 

## Dataset 

The original [Dataset](https://www.kaggle.com/datasets/datasciencedonut/current-nyc-property-sales) is sourced from Kaggle. For additional information on property addresses a secondary [PLUTO](https://catalog.data.gov/dataset/primary-land-use-tax-lot-output-pluto) dataset is used. For consumer price index [CPI](https://datahub.io/core/cpi-us) dataset is used. 

## Data Cleaning and Preprocessing 

## Exploratory Data Analysis 

## Feature Engineering 

## Model Training 

## Model Performance Evaluation 

## Deployment 

To access the streamlit app click the [Link](https://nycfamilyhousepriceprediction.streamlit.app/).

## Limitations 

## Tools Used 

## Licence 

## Contact 

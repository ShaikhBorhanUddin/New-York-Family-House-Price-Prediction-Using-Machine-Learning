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

## Project Objective 

New York City is one of the largest and most populous cities in the world, and its housing market is equally vast and complex. Building a universal property price prediction system for all property types would require massive, continuously updated datasets, high computational resources, and enterprise-grade deployment platforms. Addressing these practical constraints, this project focuses specifically on family houses, enabling accurate price prediction using limited resources and a lightweight deployment framework. The primary objective of this project is to develop and evaluate multiple machine learning regression models to predict family house prices in New York City based on historical property sales data, while accounting for location-based variation and heterogeneous housing characteristics. To ensure consistent pricing behavior and robust model performance, multi-unit apartments, condominiums, cooperatives, rental properties, and commercial real estate were intentionally excluded. Finally, the most suitable model is deployed as an interactive [Streamlit web application](https://nycfamilyhousepriceprediction.streamlit.app/) , allowing users to estimate house prices in real time through a simple and user-friendly interface. 

## Use Cases 

This project has strong real-world relevance across multiple industry domains, particularly within real estate, finance, and urban analytics. ***Real estate agencies*** and ***online property platforms*** can use the model as a decision-support tool to estimate fair/competitive market prices for family houses, enabling more accurate listings and improved buyer–seller negotiations. ***Property investors*** can use the predictions to evaluate investment opportunities, identify under or overvalued properties, and calculate potential returns based on location and property characteristics. Financial institutions such as ***banks*** and ***mortgage lenders*** can apply the model for preliminary property valuation during loan and mortgage pre-assessment, reducing reliance on time-consuming manual checkups. Additionally, ***urban planners*** and ***housing analysts*** can use aggregated prediction insights to study pricing patterns, affordability trends, and neighborhood-level market behavior, supporting data-driven planning and policy development. 

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

The following sequence is maintained consistently throughout the project to ensure a structured and reproducible machine learning pipeline:

- Upload and load the required datasets

- Perform data cleaning, preprocessing, and merge multiple datasets into a unified structure

- Derive new features through feature engineering to enhance predictive performance

- Conduct exploratory data analysis (EDA) to understand data patterns, distributions, and relationships

- Split the dataset into training and testing sets and train multiple machine learning regression models

- Analyze and compare model performance to determine the best-performing model

- Develop the application file and deploy the selected model as an interactive Streamlit web application

- Document project in repository and ensure all necessary files are uploaded

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

The primary objective of this project is to predict family house prices (up to three-family dwellings). However, the original raw datasets include all types of property transactions (such as residential buildings, office properties, factories, public establishments, and vacant land) as well as extensive property attributes and inflation data spanning from 1913 to the present. After filtering, cleaning, and feature engineering, it is important to assess how well the retained variables align with the modeling objective. Therefore, exploratory data analysis is performed on the cleaned and feature engineered dataset to understand data distributions, relationships, and potential anomalies before model training. 

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/property_location_scatterplot.png" width="42.2%" />
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/borough_barchart.png" width="56.7%" />
</p> 

The static scatter plot shows a clear and accurate spatial separation of properties across the five NYC boroughs, confirming that the latitude–longitude data aligns well with real-world geography and validating the effectiveness of the geolocation imputation process. The borough-wise distribution reveals a strong imbalance in property counts, with Queens and Brooklyn dominating the dataset, followed by Staten Island and the Bronx, while Manhattan has a notably smaller number of observations; an important consideration for model bias and representativeness. 

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/top_10_neighborhood.png" width="42.2%" />
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/zipcode_distribution.png" width="56.7%" />
</p> 

The repeated borough distribution plot reinforces this imbalance and highlights that outer boroughs contribute most of the training data, which may influence learned price patterns. The ZIP code distribution indicates that property records are heavily concentrated within specific ZIP code ranges (11000 to 11500), suggesting localized clustering of sales activity and underscoring the importance of spatial features (ZIP code, borough, latitude, longitude) in capturing neighborhood-level price variations during model training. 
 
<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/built_year_map.png" width="41.8%" />
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/year_built_barchart.png" width="57.5%" />
</p> 

The spatial distribution by year-built range shows that properties constructed between 1901 and 1950 dominate the urban landscape across all boroughs, reflecting New York City’s major early 20th century development phase. This pattern is reinforced by the year-built distribution bar chart, where the 1901–1950 era overwhelmingly outweighs other periods, followed by post-1950 constructions, while pre-1850 buildings form only a negligible share. The geographic spread of newer properties (2001–2024) appears more scattered and limited, indicating relatively modest recent residential expansion compared to historical development. 

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/sale_year.png" width="49%" />
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/sale_month.png" width="49%" />
</p> 

The sales-by-year distribution highlights clear temporal fluctuations in market activity, with higher transaction volumes in the early 2000s, a noticeable decline following the 2008 financial crisis, and a gradual recovery peaking again around 2021–2022 before tapering slightly. Meanwhile, the sales-by-month distribution reveals a strong seasonal trend, with transactions increasing from spring, peaking in summer (especially August) and remaining relatively high into early autumn, while winter months consistently record lower activity. Together, these trends suggest that both macroeconomic conditions and seasonality play a significant role in housing market dynamics. 

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/building_class_category_description.png" width="45.7%" />
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/building_class_at_time_of_sale.png" width="53.3%" />
</p> 

The building class category description bar chart shows a clear dominance of smaller residential buildings, particularly one- and two-family properties. One-family homes and dwellings account for the largest share of transactions, followed by two-family houses, while three-family properties appear much less frequently. This indicates a housing market that is strongly oriented toward low-density residential use rather than large multi-unit developments. The building class at time of sale bar chart reinforces this pattern, with categories such as A1 (two-story detached single-family homes), A2 (single-family bungalows), B1 (two-family attached), B2 (two-family detached), and B3 (converted single-family houses) comprising a substantial proportion of sales. The steep decline in transaction counts across other building classes further highlights that market activity is highly concentrated within a limited set of conventional residential property types. 

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/top_10_land_square_feet.png" width="49%" />
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/top_10_gross_square_feet.png" width="49%" />
</p> 

Land and gross square footage distributions further support the idea of standardized, moderately sized properties. Land square footage clusters strongly around a few common sizes, with lots around 2,000, 2,500, 3,000, and 4,000 square feet appearing most frequently, implying zoning or development patterns that favor repeated land sizes. Gross square footage shows a similar concentration, with peaks around 1,600 to 2,400 square feet, aligning well with typical single- and two-family homes. The presence of zero or very small gross square footage values likely reflects special property records rather than true building size. The visuals paint a picture of a fairly uniform residential market, where standard home types and sizes dominate both the physical landscape and transaction activity. 

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/sale_price_distribution.png" width="54.2%" />
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/sales_price_map.png" width="44.9%" />
</p> 

The adjusted sale price distribution indicates transaction volume peaking in the mid-range, particularly between roughly $400k and $1M, with the highest concentration around the $600k–$800k range. Below $200k and above $1.5M, counts fall off steadily, indicating that both very low-priced and very high-priced sales are relatively uncommon. Ultra-high-value transactions above $2M exist but represent a small fraction of overall activity, tapering sharply beyond $5M. This pattern is echoed spatially in the map. Most properties appear in lighter shades across the outer boroughs (especially Queens, Brooklyn outside the waterfront, and Staten Island) reflecting the dominance of mid-priced transactions. In contrast, the darkest, highest-value clusters are tightly concentrated in Manhattan, particularly along central corridors and waterfront areas, with smaller pockets in northwestern Brooklyn. Together, the visuals highlight a market where mid-range prices dominate transaction volume, while extreme property values are both rare and highly localized, underscoring the strong role of geography in shaping price disparities. 

## Model Training 

For model training, several columns were excluded as they do not add predictive value given the project’s focus on family house price prediction. Columns like `TAX CLASS AT PRESENT`, `BUILDING CLASS AT PRESENT`, and `TAX CLAAS AT TIME OF SALE` were removed, as only residential family homes (up to three families) are considered and these administrative attributes are unlikely to influence sale prices meaningfully. The `TOTAL UNITS` column was also dropped because it is simply the sum of residential and commercial units, both of which are already represented separately, making it redundant. Since the target variable is the inflation-adjusted sale price, the original `SALE PRICE` and `CPI` columns were removed after deriving the adjusted value. The `SALE DATE` column was dropped because it was decomposed into `SALE YEAR` and `SALE MONTH`, and retaining the exact transaction date could introduce unnecessary high cardinality and potential data leakage. Since the combination of `NEIGHBORHOOD`, `BLOCK`, and `LOT` already captures property location information, the `latitude` and `longitude` columns were removed to avoid feature redundancy. 

Categorical variables were handled using one-hot encoding to make them suitable for machine learning algorithms. Specifically, `NEIGHBORHOOD`, `BUILDING CLASS AT TIME OF SALE`, and `BUILDING CLASS CATEGORY DESCRIPTION` were encoded, resulting in a high-dimensional but fully numerical feature space. After preprocessing, the final dataset was split into training and testing sets, with the training data having a shape of (439248 : 283) and the test data having a shape of (109812 : 283), indicating a large-scale supervised learning setup. 

Three different regression models were trained and compared to capture both linear and non-linear relationships in the data. The model configurations are described below. 

| Model Name    | Implementation                          | Key Parameters Used                                                                                       | Training Time |
|---------------|------------------------------------------|-------------------------------------------------------------------------------------------------------------|---------------|
| XGBoost       | `xgb.XGBRegressor`                      | `objective='reg:squarederror'`, `n_estimators=100`, `learning_rate=0.1`, `max_depth=5`, `random_state=42` |   5 Seconds            |
| ElasticNet    | `Pipeline(StandardScaler → ElasticNet)` | `random_state=42`                                                                                         |  5 Seconds             |
| Random Forest | `RandomForestRegressor`                 | `n_estimators=100`, `random_state=42`, `n_jobs=-1`                                                        |    22 Minutes           | 

## Model Performance Evaluation 

The performance metrics of all three tested models are presented in this section, evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²). 

| Model Name     | MAE        | MSE              | RMSE       | R²  |
|----------------|------------|------------------|------------|-----|
| XGBoost        | 241,319.42 | 537,487,507,456  | 733,135.40 | 0.69 |
| ElasticNet     | 288,835.18 | 955,445,796,725  | 977,469.08 | 0.45 |
| Random Forest  | 208,127.17 | 469,900,136,496  | 685,492.62 | 0.73 | 

Based on the evaluation metrics, **Random Forest** is the best-performing model in this project. It achieves the lowest MAE (208,127.17) and lowest RMSE (685,492.62), indicating more accurate price predictions with smaller average and squared errors. In addition, it records the highest R² score (0.73), meaning it explains the largest proportion of variance in adjusted house prices compared to XGBoost and ElasticNet. 

To assess how accurately the models predict house prices, predictions for 40 randomly selected samples from the test dataset were compared against the actual values. However, based on this comparison, the Random Forest model did not demonstrate a clear or consistent superiority over the other models. 

![Dashboard](https://github.com/ShaikhBorhanUddin/New-York-Property-Price-Prediction-Using-Machine-Learning/blob/main/Assets/actual_vs_predicted.png?raw=true) 

Plots with a locally weighted regression (LOESS) curve for each model was also generated. This type of curve is more adaptive than a simple polynomial and can better reflect the native trend of how each model's predictions relate to the actual values without imposing a fixed functional form. This gives a good visual sense of the general agreement between actual and predicted values for each model. 

![Dashboard](https://github.com/ShaikhBorhanUddin/New-York-Property-Price-Prediction-Using-Machine-Learning/blob/main/Assets/LOESS.png?raw=true) 

Both XGBoost and Random Forest demonstrate superior predictive performance over ElasticNet, as indicated by the LOESS curve. 

## Deployment 

Although Random Forest achieved the best performance among the three models, its large size (3.66 GB) made it unsuitable for deployment in a lightweight framework such as Streamlit Web. Consequently, the second-best performing model, XGBoost, was selected for deployment due to its significantly smaller size (275 KB). In addition to the trained model, four additional .pkl files were required to support consistent preprocessing and inference during deployment. 

- To avoid feature name mismatches at inference time, the complete list of feature names was saved as `feature_names.pkl` file and loaded as an artifact within the application. 

- The unique values for the categorical features `NEIGHBORHOOD`, `BUILDING CLASS AT TIME OF SALE`, and `BUILDING CLASS CATEGORY DESCRIPTION` were extracted from the original DataFrame (251, 18, and 6 unique values respectively). These values were stored in a dictionary and serialized into a `unique_categorical_values.pkl` file, which is loaded as an artifact in application file during deployment. This ensures that user inputs are constrained to valid, NYC-realistic categories, preventing arbitrary or unrealistic text entries and maintaining consistency with the training data. 

- To support dynamic, dependency-aware filtering in the Streamlit application, structured mappings were constructed between `BOROUGH`, `NEIGHBORHOOD`, and `ZIP CODE`. Four lookup dictionaries were generated: **borough-to-neighborhoods**, **borough-to-zipcodes**, **neighborhood-to-zipcodes**, and **zipcode-to-borough** (under the dataset constraint that each ZIP code maps to a single borough). These lookup tables were merged into a unified artifact, `combined_location_mapping.pkl`, which is loaded at runtime by application to enforce valid hierarchical relationships and ensure consistent, data-driven user input across all location fields. 

- The average latitude and longitude were computed for each unique combination of `BOROUGH`, `NEIGHBORHOOD`, and `ZIP CODE` from the original dataset and stored as a structured dictionary. This mapping was serialized and saved as `location_coordinates_mapping.pkl`, which is loaded by the application to dynamically resolve geographic coordinates based on user-selected location inputs and update the map visualization in real time.

To access the streamlit app click the [Link](https://nycfamilyhousepriceprediction.streamlit.app/). Users can enter property details such as location, building characteristics, size, and sale information (including borough, neighborhood, ZIP code, unit counts, square footage, construction year, and building class). Once the information is submitted, the application instantly predicts the current market price of the property. 

![Dashboard](https://github.com/ShaikhBorhanUddin/New-York-Property-Price-Prediction-Using-Machine-Learning/blob/main/Assets/app_screenshot.png?raw=true)  

## Limitations 

Despite careful model development and deployment, several data and system level limitations should be considered when interpreting the predictions. 

Since the model was trained exclusively on family house sale data, the dataset represents only a subset of all properties within each neighborhood. Properties that were never sold (e.g., inherited or endowed properties) are not included in the training data. In addition, the model assumes that geographic identifiers such as `BOROUGH`, `NEIGHBORHOOD`, and `ZIP CODE` are static and consistently defined over time. In reality, neighborhood boundaries, zoning regulations, land use classifications, and administrative definitions may evolve due to urban development, rezoning, or gentrification effects. These temporal and structural changes are not captured in the dataset. As a result of both incomplete property coverage and static geographic assumptions, dynamic filtering for `BLOCK` and `LOT` could not be implemented in the Streamlit application. Such functionality would only be feasible if a complete, time-aware inventory of all properties within each neighborhood were available. 

A Folium-based property location and sale price heatmap over Google Maps would provide significantly richer spatial insights compared to simple geolocation markers in the application. However, rendering such visualizations for large datasets is both time and memory intensive, leading to slow load times and a degraded user experience. The visualizations shown below illustrate this limitation, even when plotting a reduced sample of 50,000 data points. 

<p align="center">
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/50k_sample_scatterplot.png" width="44.2%" />
  <img src="https://github.com/ShaikhBorhanUddin/New-York-Family-House-Price-Prediction-Using-Machine-Learning/blob/main/Assets/folium_property_heatmap.png" width="54.8%" />
</p> 

For deployment, all available data points would need to be rendered to enable meaningful comparison, which would further increase computational overhead. Therefore, although these heatmap visualizations were explored and analyzed during the Exploratory Data Analysis phase, they were intentionally excluded from the deployed Streamlit application to ensure smooth responsiveness and usability. 

Finally, a key limitation of this project is the exclusion of the Hedonic Price Model (HPM), primarily due to data and computational constraints. Implementing HPM effectively requires detailed and accurate information on all property attributes as well as location-based factors (e.g., school ratings, neighborhood crime statistics, proximity to amenities), many of which were unavailable in the datasets. Additionally, integrating HPM with machine learning models would have increased the computational complexity and resource requirements, which could have made the lightweight deployment framework less efficient and more difficult to maintain. 

## Tools Used 

This project was built using the following tools and technologies:

### Programming & Development

- `Python` — Core programming language for data processing, modeling, and deployment.

- `Jupyter Notebook` — Interactive environment for exploratory data analysis, preprocessing, feature engineering, and model training.

### Machine Learning & Data Processing

- `pandas` — Data manipulation and cleaning.

- `NumPy` — Numerical computing and array operations.

- `scikit-learn` — Machine learning library used for model training, evaluation, and preprocessing.

- `XGBoost` — Gradient boosting algorithm used for high-performance regression modeling.

- `joblib / pickle` — Model serialization and saving for deployment.

### Visualization

- `Matplotlib` — Plotting and visualization for exploratory data analysis and model diagnostics.

- `Seaborn` — Enhanced statistical visualizations.

- `Folium` — Map visualization for EDA.

### Deployment & App Interface

- `VS Code` — Python framework used for coding the interactive web application.

- `Streamlit` — Python framework used to deploy the interactive web application for house price prediction.

### Version Control & Environment

- `Git` & `GitHub` — Source code versioning and project hosting. 

## Licence 

This project is licensed under the MIT License, a permissive open-source license that allows reuse, modification, and distribution with attribution. Users are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the project, provided that the original copyright and license notice are included in all copies or substantial portions of the software. 

## Contact 

If you have any questions or would like to connect, feel free to reach out. 

**Shaikh Borhan Uddin** 

📧 [`Email`](mailto:shaikhborhanuddin@gmail.com) 🔗 [`LinkedIn`](https://www.linkedin.com/in/shaikh-borhan-uddin-905566253/) 🌐 [`Portfolio`](https://github.com/ShaikhBorhanUddin) 

Feel free to fork the repository, experiment with other models, or improve deployment! 

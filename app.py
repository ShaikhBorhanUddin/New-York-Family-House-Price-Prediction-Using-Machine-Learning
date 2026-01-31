import streamlit as st
import pandas as pd
import joblib
import xgboost
import folium
from streamlit_folium import st_folium

# Page config 
st.set_page_config( page_title="NYC Family House Price Prediction", layout="wide")

# File Paths
model_path = 'Models/xgboost_model.pkl'
feature_names_path = 'Models/feature_names.pkl'
unique_categorical_values_path = 'Models/unique_categorical_values.pkl'
combined_location_mapping_path = 'Models/combined_location_mapping.pkl'
location_coordinates_mapping_path = 'Models/location_coordinates_mapping.pkl'

# Load Model & Artifacts
@st.cache_resource
def load_model():
    return joblib.load(model_path)

@st.cache_resource
def load_artifacts():
    return (
        joblib.load(feature_names_path),
        joblib.load(unique_categorical_values_path),
        joblib.load(combined_location_mapping_path),
        joblib.load(location_coordinates_mapping_path),
    )

model = load_model()
feature_names, unique_categorical_values, combined_location_mapping, location_coordinates_mapping = load_artifacts()

# Mappings & Defaults
borough_names_map = {
    1: 'Manhattan',
    2: 'Bronx',
    3: 'Brooklyn',
    4: 'Queens',
    5: 'Staten Island'
}

initial_borough_id = 1
initial_neighborhood = 'ALPHABET CITY'
initial_zip_code = 10009
initial_latitude = 40.7128
initial_longitude = -74.0060

# Helper Functions
def update_lat_lon():
    key = (
        st.session_state.selected_borough_id,
        st.session_state.selected_neighborhood_name,
        st.session_state.selected_zip_code
    )
    if key in location_coordinates_mapping:
        st.session_state.selected_latitude, st.session_state.selected_longitude = location_coordinates_mapping[key]
    else:
        st.session_state.selected_latitude = initial_latitude
        st.session_state.selected_longitude = initial_longitude


def get_select_box_index(options, value):
    try:
        return options.index(value)
    except ValueError:
        return 0


def on_borough_change():
    st.session_state.selected_borough_id = st.session_state.borough_select
    neighborhoods = combined_location_mapping['borough_to_neighborhoods'].get(
        st.session_state.selected_borough_id, []
    )
    st.session_state.selected_neighborhood_name = neighborhoods[0] if neighborhoods else ''

    zipcodes = sorted(
        set(combined_location_mapping['borough_to_zipcodes'].get(st.session_state.selected_borough_id, [])) &
        set(combined_location_mapping['neighborhood_to_zipcodes'].get(st.session_state.selected_neighborhood_name, []))
    )
    st.session_state.selected_zip_code = zipcodes[0] if zipcodes else 0
    update_lat_lon()


def on_neighborhood_change():
    st.session_state.selected_neighborhood_name = st.session_state.neighborhood_select
    zipcodes = sorted(
        set(combined_location_mapping['borough_to_zipcodes'].get(st.session_state.selected_borough_id, [])) &
        set(combined_location_mapping['neighborhood_to_zipcodes'].get(st.session_state.selected_neighborhood_name, []))
    )
    st.session_state.selected_zip_code = zipcodes[0] if zipcodes else 0
    update_lat_lon()


def on_zip_code_change():
    st.session_state.selected_zip_code = st.session_state.zip_code_select
    update_lat_lon()

# Session State Init
if 'selected_borough_id' not in st.session_state:
    st.session_state.selected_borough_id = initial_borough_id

if 'selected_neighborhood_name' not in st.session_state:
    st.session_state.selected_neighborhood_name = initial_neighborhood

if 'selected_zip_code' not in st.session_state:
    st.session_state.selected_zip_code = initial_zip_code

if 'selected_latitude' not in st.session_state:
    update_lat_lon()

# App Title
st.title("NYC Family House Price Prediction")
st.write("Enter the property details to predict its current sale price.")

# Layout: 3 Columns
col1, col2, col3 = st.columns([1.1, 1.2, 1])

# Column 1: Location + Block/Lot + Building
with col1:
    st.header("Location Details")

    st.selectbox(
        "BOROUGH",
        options=list(borough_names_map.keys()),
        index=get_select_box_index(list(borough_names_map.keys()), st.session_state.selected_borough_id),
        format_func=lambda x: borough_names_map[x],
        key="borough_select",
        on_change=on_borough_change
    )

    neighborhoods = combined_location_mapping['borough_to_neighborhoods'].get(
        st.session_state.selected_borough_id, []
    )

    st.selectbox(
        "NEIGHBORHOOD",
        options=neighborhoods,
        index=get_select_box_index(neighborhoods, st.session_state.selected_neighborhood_name),
        key="neighborhood_select",
        on_change=on_neighborhood_change
    )

    zipcodes = sorted(
        set(combined_location_mapping['borough_to_zipcodes'].get(st.session_state.selected_borough_id, [])) &
        set(combined_location_mapping['neighborhood_to_zipcodes'].get(st.session_state.selected_neighborhood_name, []))
    )

    st.selectbox(
        "ZIP CODE",
        options=zipcodes,
        index=get_select_box_index(zipcodes, st.session_state.selected_zip_code),
        key="zip_code_select",
        on_change=on_zip_code_change
    )

    block = st.number_input("BLOCK", min_value=1, max_value=16350, value=374)
    lot = st.number_input("LOT", min_value=1, max_value=4336, value=46)

    st.header("Building Characteristics")

    selected_building_class = st.selectbox(
        "BUILDING CLASS AT TIME OF SALE",
        unique_categorical_values['BUILDING CLASS AT TIME OF SALE'],
        index=get_select_box_index(
            unique_categorical_values['BUILDING CLASS AT TIME OF SALE'], 'A4'
        )
    )

    selected_building_category = st.selectbox(
        "BUILDING CLASS CATEGORY DESCRIPTION",
        unique_categorical_values['BUILDING CLASS CATEGORY DESCRIPTION'],
        index=get_select_box_index(
            unique_categorical_values['BUILDING CLASS CATEGORY DESCRIPTION'], 'ONE FAMILY DWELLINGS'
        )
    )

# Column 2: Property Details
with col2:
    st.header("Property Details")

    residential_units = st.number_input("RESIDENTIAL UNITS", min_value=0, value=1)
    commercial_units = st.number_input("COMMERCIAL UNITS", min_value=0, value=0)

    land_square_feet = st.number_input("LAND SQUARE FEET", min_value=0.0, value=2116.0)
    gross_square_feet = st.number_input("GROSS SQUARE FEET", min_value=0.0, value=4400.0)

    year_built = st.number_input("YEAR BUILT", 1700, 2024, 1900)
    sale_year = st.number_input("SALE YEAR", 2000, 2024, 2022)
    sale_month = st.number_input("SALE MONTH", 1, 12, 9)

# Column 3: Coordinates + Map
with col3:
    st.header("Property Location (for Map Visualization)")

    latitude = st.number_input(
        "LATITUDE",
        min_value=-90.0,
        max_value=90.0,
        value=float(st.session_state.selected_latitude),
        format="%.6f"
    )

    longitude = st.number_input(
        "LONGITUDE",
        min_value=-180.0,
        max_value=180.0,
        value=float(st.session_state.selected_longitude),
        format="%.6f"
    )

    st.session_state.selected_latitude = latitude
    st.session_state.selected_longitude = longitude

    st.subheader("Property Location Map")

    m = folium.Map(location=[latitude, longitude], zoom_start=15)
    folium.Marker([latitude, longitude]).add_to(m)
    st_folium(m, width=None, height=380)

# Prediction
if st.button("Predict Sale Price"):
    input_data = {
        'BOROUGH': st.session_state.selected_borough_id,
        'BLOCK': block,
        'LOT': lot,
        'ZIP CODE': st.session_state.selected_zip_code,
        'RESIDENTIAL UNITS': residential_units,
        'COMMERCIAL UNITS': commercial_units,
        'LAND SQUARE FEET': land_square_feet,
        'GROSS SQUARE FEET': gross_square_feet,
        'YEAR BUILT': year_built,
        'SALE YEAR': sale_year,
        'SALE MONTH': sale_month,
        'NEIGHBORHOOD': st.session_state.selected_neighborhood_name,
        'BUILDING CLASS AT TIME OF SALE': selected_building_class,
        'BUILDING CLASS CATEGORY DESCRIPTION': selected_building_category
    }

    df = pd.DataFrame([input_data])
    ohe_cols = [
        'NEIGHBORHOOD',
        'BUILDING CLASS AT TIME OF SALE',
        'BUILDING CLASS CATEGORY DESCRIPTION'
    ]
    df_ohe = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    final_df = pd.DataFrame(0, index=[0], columns=feature_names)
    for col in df_ohe.columns:
        if col in final_df.columns:
            final_df[col] = df_ohe[col]

    prediction = model.predict(final_df)[0]
    st.success(f"Predicted Current House Price: ${prediction:,.2f}")

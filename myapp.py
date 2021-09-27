import streamlit as st
import pandas as pd
import seaborn as sn
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
 

st.write("""
# US Foreign Aid Disbursement Amount Prediction App
""")
st.write('---')

# Read the csv file into a pandas DataFrame
df = pd.read_csv('x_y_merged.csv')

X = df[['year', 'cancer',
        'air_pollution_death', 'alchol_abuse', 'basic_sanitation',
        'drinking_water', 'hand_wash', 'fuel_tech', 'crude_suicide',
       'tuberculosis', 'NTDs', 'doctors', 'poisoning', 'unsafe_wash',
       'tobacco', 'UHC_coverage', 'life_expectancy', 'UHC_data_access',
       'population']]

Y = df['disbursement_amount'].values.ravel()

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('User Input Features')

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

else:

    def user_input_features():
        year = st.sidebar.number_input('year', float(X.year.min()), float(X.year.max()), float(X.year.mean()))
        cancer = st.sidebar.number_input('cancer', float(X.cancer.min()), float(X.cancer.max()), float(X.cancer.mean()))
        air_pollution_death = st.sidebar.number_input('air_pollution_death', float(X.air_pollution_death.min()), float(X.air_pollution_death.max()), float(X.air_pollution_death.mean()))
        alchol_abuse = st.sidebar.number_input('alchol_abuse', float(X.alchol_abuse.min()), float(X.alchol_abuse.max()), float(X.alchol_abuse.mean()))
        basic_sanitation = st.sidebar.number_input('basic_sanitation', float(X.basic_sanitation.min()), float(X.basic_sanitation.max()), float(X.basic_sanitation.mean()))
        drinking_water = st.sidebar.number_input('drinking_water', float(X.drinking_water.min()), float(X.drinking_water.max()), float(X.drinking_water.mean()))
        hand_wash = st.sidebar.number_input('hand_wash', float(X.hand_wash.min()), float(X.hand_wash.max()), float(X.hand_wash.mean()))
        fuel_tech = st.sidebar.number_input('fuel_tech', float(X.fuel_tech.min()), float(X.fuel_tech.max()), float(X.fuel_tech.mean()))
        crude_suicide = st.sidebar.number_input('crude_suicide', float(X.crude_suicide.min()), float(X.crude_suicide.max()), float(X.crude_suicide.mean()))
        tuberculosis = st.sidebar.number_input('tuberculosis', float(X.tuberculosis.min()), float(X.tuberculosis.max()), float(X.tuberculosis.mean()))
        NTDs = st.sidebar.number_input('NTDs', float(X.NTDs.min()), float(X.NTDs.max()), float(X.NTDs.mean()))
        doctors = st.sidebar.number_input('doctors', float(X.doctors.min()), float(X.doctors.max()), float(X.doctors.mean()))
        poisoning = st.sidebar.number_input('poisoning', float(X.poisoning.min()), float(X.poisoning.max()), float(X.poisoning.mean()))
        unsafe_wash = st.sidebar.number_input('unsafe_wash', float(X.unsafe_wash.min()), float(X.unsafe_wash.max()), float(X.unsafe_wash.mean()))
        tobacco = st.sidebar.number_input('tobacco', float(X.tobacco.min()), float(X.tobacco.max()), float(X.tobacco.mean()))
        UHC_coverage = st.sidebar.number_input('UHC_coverage', float(X.UHC_coverage.min()), float(X.UHC_coverage.max()), float(X.UHC_coverage.mean()))
        life_expectancy = st.sidebar.number_input('life_expectancy', float(X.life_expectancy.min()), float(X.life_expectancy.max()), float(X.life_expectancy.mean()))
        UHC_data_access = st.sidebar.number_input('UHC_data_access', float(X.UHC_data_access.min()), float(X.UHC_data_access.max()), float(X.UHC_data_access.mean()))
        population = st.sidebar.number_input('population', float(X.population.min()), float(X.population.max()), float(X.population.mean()))

        data = {'year': year,
                'cancer': cancer,
                'air_pollution_death': air_pollution_death,
                'alchol_abuse': alchol_abuse,
                'basic_sanitation': basic_sanitation,
                'drinking_water': drinking_water,
                'hand_wash': hand_wash,
                'fuel_tech': fuel_tech,
                'crude_suicide': crude_suicide,
                'tuberculosis': tuberculosis,
                'NTDs': NTDs,
                'doctors': doctors,
                'poisoning': poisoning,
                'unsafe_wash': unsafe_wash,
                'tobacco': tobacco,
                'UHC_coverage': UHC_coverage,
                'life_expectancy': life_expectancy,
                'UHC_data_access': UHC_data_access,
                'population': population
                }
                
        features = pd.DataFrame(data, index=[0])
        return features

    df2 = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df2)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)
# Apply Model to Make Prediction
prediction = model.predict(df2)

st.header('Prediction of MEDV')
st.write(prediction)
st.write('---')
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sn
import pickle
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
 

st.write("""
# Foreign Aid Needed Priority Level Prediction App
""")
st.write('---')

# Read the csv file into a pandas DataFrame
df_aid_raw = pd.read_csv('aid_clustering.csv')
aid = df_aid_raw.drop(columns=['k_labels'])

X = df_aid_raw[['child_mort', 'exports',
        'health', 'imports', 'income',
        'inflation', 'life_expec', 'total_fer', 'gdpp']]

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('User Input Features')

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

else:

    def user_input_features():
        child_mort = st.sidebar.number_input('child_mort', float(X.child_mort.min()), float(X.child_mort.max()), float(X.child_mort.mean()))
        exports = st.sidebar.number_input('exports', float(X.exports.min()), float(X.exports.max()), float(X.exports.mean()))
        health = st.sidebar.number_input('health', float(X.health.min()), float(X.health.max()), float(X.health.mean()))
        imports = st.sidebar.number_input('imports', float(X.imports.min()), float(X.imports.max()), float(X.imports.mean()))
        income = st.sidebar.number_input('income', float(X.income.min()), float(X.income.max()), float(X.income.mean()))
        inflation = st.sidebar.number_input('inflation', float(X.inflation.min()), float(X.inflation.max()), float(X.inflation.mean()))
        life_expec = st.sidebar.number_input('life_expec', float(X.life_expec.min()), float(X.life_expec.max()), float(X.life_expec.mean()))
        total_fer = st.sidebar.number_input('total_fer', float(X.total_fer.min()), float(X.total_fer.max()), float(X.total_fer.mean()))
        gdpp = st.sidebar.number_input('gdpp', float(X.gdpp.min()), float(X.gdpp.max()), float(X.gdpp.mean()))

        data = {'child_mort': child_mort,
                'exports': exports,
                'health': health,
                'imports': imports,
                'income': income,
                'inflation': inflation,
                'life_expec': life_expec,
                'total_fer': total_fer,
                'gdpp': gdpp
                }
                
        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input_features()


df = pd.concat([input_df,aid],axis=0)

# Main Panel

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(input_df)

# Reads in saved classification model
load_clf = pickle.load(open('aid_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(input_df)
prediction_proba = load_clf.predict_proba(input_df)


st.subheader('Prediction')
aid_priority = np.array(['Aid needed priority-1','Aid needed priority-2','Aid needed priority-3','No Aid needed'])
st.write(aid_priority[prediction])


import streamlit as st
import pandas as pd


def feature_engineering(data_analysis):
 
    st.title("Feature Engineering & Selection")
    st.write("This section provides insights into feature engineering and selection.")

    st.markdown("**1. Current Feature List**")
    st.write(data_analysis.columns.tolist())

    st.markdown("**2. Missing Values**")
    st.write(data_analysis.isnull().sum())

    st.markdown("**3. Feature Selection**")
    selected_features = ['Age', 'BUN', 'Creatinine_Level', 'Urine_Output', 'Diabetes', 'Hypertension', 'GFR']
    st.write("Selected for modeling:", selected_features)

    st.markdown("**4. Feature Engineering**")
    data_eng = data_analysis.copy()
    data_eng['BUN_Category'] = pd.cut(data_eng['BUN'], bins=[0, 20, 40, 60, 80, 100], 
                                        labels=['Low', 'Medium', 'High', 'Very High', 'Extreme'], right=False)
    data_eng['Creatinine_Category'] = pd.cut(data_eng['Creatinine_Level'], bins=[0, 1, 2, 3, 4, 5], 
                                                labels=['Normal', 'Mild', 'Moderate', 'Severe', 'Critical'], right=False)
    data_eng['GFR_category'] = pd.cut(data_eng['GFR'], bins=[0, 15, 30, 45, 60, 75, 90, 120],
                                        labels=['Very Low', 'Low', 'Moderate', 'Mild', 'Normal', 'High', 'Very High'], right=False)
    data_eng['Age_Group'] = pd.cut(data_eng['Age'], bins=[0, 20, 40, 60, 80, 100], 
                                    labels=['0–20', '21–40', '41–60', '61–80', '81+'], right=False)

    engineered_features = ['Age_Group', 'BUN_Category', 'Creatinine_Category', 'GFR_category']
    st.write("Engineered:", engineered_features)

    dropped_features = ['BUN', 'Creatinine_Level', 'Urine_Output', 'Dialysis_Needed', 'Age', 'GFR']
    data_eng.drop(columns=dropped_features, inplace=True)

    st.markdown("**5. Final Feature List**")
    st.write(data_eng.columns.tolist())

    st.markdown("**Preview of Engineered Data**")
    st.dataframe(data_eng.head())

    # Store for next page
    st.session_state['data_for_model'] = data_eng.copy()

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import your custom modules (make sure these files exist)
try:
    from eda import eda
    from feature import feature_engineering
    from model import model_training
    from prediction import model_prediction
except ImportError as e:
    st.error(f"Import error: {e}. Please ensure all module files exist.")

st.set_page_config(page_title="Kidney Disease Prediction Dashboard", layout="wide")

st.title("Kidney Disease Prediction Dashboard")

# Load the dataset with error handling
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('kidney_disease_dataset.csv', encoding='utf-8')
        return data
    except FileNotFoundError:
        st.error("Dataset file 'kidney_disease_dataset.csv' not found. Please ensure the file is in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

data_kidney = load_data()

if data_kidney is not None:
    st.subheader("Kidney Disease Dataset Overview")
    
    st.write("This dataset contains information about patients with kidney disease. The features include various medical attributes and the target variable indicates whether the patient has kidney disease.")
    
    # Displaying the dataset
    st.subheader("Dataset Preview")
    overview = st.checkbox("Show dataset overview")
    if overview:
        st.write("Here is a preview of the dataset:")
        st.dataframe(data_kidney)
    
    # Display the shape of the dataset
    st.subheader("Dataset Shape")
    st.write(f"The dataset contains {data_kidney.shape[0]} rows and {data_kidney.shape[1]} columns.")
    
    # Displaying the columns of the dataset
    st.subheader("Dataset Columns")
    st.write("The dataset contains the following columns:")
    st.write(data_kidney.columns.tolist())
    
    # Display the datatypes of the columns
    st.subheader("Data Types of Columns")
    st.write("The data types of the columns are as follows:")
    st.write(data_kidney.dtypes)
    
    # Displaying the summary statistics of the dataset
    st.subheader("Summary Statistics")
    summary_stats = st.checkbox("Show summary statistics")
    if summary_stats:
        st.write("Here are the summary statistics of the dataset:")
        st.dataframe(data_kidney.describe())
    
    # Create a copy for analysis
    data_analysis = data_kidney.copy()
    
    # Map categorical variables (check if columns exist first)
    mapping_dict = {}
    if 'CKD_Status' in data_analysis.columns:
        mapping_dict['CKD_Status'] = {1: 'CKD', 0: 'Not CKD'}
    if 'Dialysis_Needed' in data_analysis.columns:
        mapping_dict['Dialysis_Needed'] = {1: 'Yes', 0: 'No'}
    if 'Diabetes' in data_analysis.columns:
        mapping_dict['Diabetes'] = {1: 'Yes', 0: 'No'}
    if 'Hypertension' in data_analysis.columns:
        mapping_dict['Hypertension'] = {1: 'Yes', 0: 'No'}
    
    if mapping_dict:
        data_analysis.replace(mapping_dict, inplace=True)
    
    st.subheader("Data Analysis")
    st.write("This section will provide insights into the dataset through various visualizations and analysis")
    
    # Show head of the mapped dataset
    st.markdown("**Review of dataset after mapping categorical variables**")
    st.dataframe(data_analysis.head())
    
    # Define color palettes
    custom_palette = ["#1B2558", "#856506", "#0b398a", "#f8b71f", "#0D048F", "#bb931d", "#0B053A", "#cf9202"]
    blue_palette = ["#358ff7", "#125ee0", "#0E0681", "#0B053A"]
    orange_palette = ["#856506", "#bb931d", "#e0a009", "#e6b951"]
    
    # Statistical analysis 
    st.markdown("**Statistical Analysis**")
    
    # Check which target variables exist
    available_targets = []
    if 'CKD_Status' in data_analysis.columns:
        available_targets.append('CKD_Status')
    if 'Dialysis_Needed' in data_analysis.columns:
        available_targets.append('Dialysis_Needed')
    
    if available_targets:
        target_options = st.multiselect("Select Target Variable for Analysis", available_targets, default=None)
        st.write("Selected Target Variable(s):", target_options)
        
        if target_options:
            col1, col2 = st.columns(2)
            with col1:
                for target in target_options:
                    st.subheader(f"Statistical Analysis for {target}")
                    st.write("Value counts:")
                    st.write(data_analysis[target].value_counts())
                    st.write("Proportions:")
                    st.write(data_analysis[target].value_counts(normalize=True))
                    
                    # Only show groupby describe if there are numeric columns
                    numeric_cols = data_analysis.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        st.write("Grouped statistics:")
                        st.write(data_analysis.groupby([target])[numeric_cols].describe().transpose())
                    
                    # Plotting the distribution of the target variable
                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.countplot(x=target, data=data_analysis, palette=custom_palette[:2], ax=ax)
                    ax.set_title(f'Distribution of {target}')
                    ax.set_xlabel(target)
                    ax.set_ylabel('Count')
                    st.pyplot(fig)
                    plt.close()
    
    st.subheader("General Analysis with Visualizations")
    st.write("This section provides general analysis of the dataset with visualizations.")
    
    col1, col2 = st.columns(2)
    
    # CKD Status distribution
    if 'CKD_Status' in data_analysis.columns:
        with col1:
            st.markdown("**Distribution of CKD Status**")
            sizes = data_analysis['CKD_Status'].value_counts().values
            labels = data_analysis['CKD_Status'].value_counts().index
            
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(sizes, labels=labels, colors=custom_palette, startangle=90, 
                   wedgeprops={'width': 0.4}, autopct='%1.1f%%')
            ax.set_title('Distribution of CKD Status')
            st.pyplot(fig)
            plt.close()
    
    # Dialysis Needed distribution
    if 'Dialysis_Needed' in data_analysis.columns:
        with col2:
            st.markdown("**Distribution of Dialysis Needed**")
            sizes = data_analysis['Dialysis_Needed'].value_counts().values
            labels = data_analysis['Dialysis_Needed'].value_counts().index
            
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(sizes, labels=labels, colors=custom_palette[2:4], startangle=90, 
                   wedgeprops={'width': 0.4}, autopct='%1.1f%%')
            ax.set_title('Distribution of Dialysis Needed')
            st.pyplot(fig)
            plt.close()
    
    col1, col2 = st.columns(2)
    
    # Dialysis by CKD Status
    if 'Dialysis_Needed' in data_analysis.columns and 'CKD_Status' in data_analysis.columns:
        with col1:
            st.markdown("**Dialysis Needed by CKD Status**")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.countplot(data=data_analysis, x="Dialysis_Needed", hue="CKD_Status", 
                         palette=custom_palette, ax=ax)
            ax.set_title("Dialysis Needed by CKD Status")
            ax.set_xlabel("Dialysis Needed")
            ax.set_ylabel("Count")
            ax.legend(title="CKD Status")
            st.pyplot(fig)
            plt.close()
    
    # Age groups distribution
    if 'Age' in data_analysis.columns:
        with col2:
            bins = [0, 18, 35, 50, 65, 100]
            labels = ['0-18', '19-35', '36-50', '51-65', '66+']
            data_analysis['Age_Group'] = pd.cut(data_analysis['Age'], bins=bins, labels=labels, right=False)
            st.markdown("**Distribution of Age Groups**")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.countplot(data=data_analysis, x='Age_Group', palette=blue_palette, ax=ax)
            ax.set_title('Distribution of Age Groups')
            ax.set_xlabel('Age Group')
            ax.set_ylabel('Count')
            plt.setp(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)
            plt.close()
    
    # Correlation analysis
    numeric_data = data_analysis.select_dtypes(include=[np.number])
    if len(numeric_data.columns) > 1:
        col3, col4 = st.columns(2)
        with col3:
            cmap = sns.color_palette("blend:#0E0681,#856506", as_cmap=True)
            st.markdown("**Correlation Analysis**")
            correlation_matrix = numeric_data.corr()
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap=cmap, fmt='.2f', 
                       linewidths=0.5, ax=ax)
            ax.set_title('Correlation Matrix')
            st.pyplot(fig)
            plt.close()
    
    # Navigation section
    st.sidebar.title("Navigation")
    # Define pages with error handling
    def safe_eda():
        try:
            eda(data_analysis=data_analysis)
        except NameError:
            st.error("EDA function not available. Please ensure the 'eda' module is properly imported.")
        except Exception as e:
            st.error(f"Error in EDA: {e}")
    
    def safe_feature_engineering():
        try:
            feature_engineering(data_analysis=data_analysis)
        except NameError:
            st.error("Feature engineering function not available. Please ensure the 'feature' module is properly imported.")
        except Exception as e:
            st.error(f"Error in feature engineering: {e}")
    
    def safe_model_training():
        try:
            model_training(data_analysis=data_analysis)
        except NameError:
            st.error("Model training function not available. Please ensure the 'model' module is properly imported.")
        except Exception as e:
            st.error(f"Error in model training: {e}")
    
    def safe_model_prediction():
        try:
            model_prediction()
        except NameError:
            st.error("Model prediction function not available. Please ensure the 'prediction' module is properly imported.")
        except Exception as e:
            st.error(f"Error in model prediction: {e}")
    
    # Create navigation
    analysis_option = st.sidebar.selectbox(
        "Choose an analysis to perform:",
        ["Select an option", "Exploratory Data Analysis (EDA)", 
         "Feature Engineering & Selection", "Model Training & Evaluation", "Model Prediction"]
    )
    
    if analysis_option == "Exploratory Data Analysis (EDA)":
        safe_eda()
    elif analysis_option == "Feature Engineering & Selection":
        safe_feature_engineering()
    elif analysis_option == "Model Training & Evaluation":
        safe_model_training()
    elif analysis_option == "Model Prediction":
        safe_model_prediction()

else:
    st.stop()
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def eda(data_analysis):
    custom_palette = ["#1B2558", "#856506", "#0b398a", "#f8b71f", "#0D048F", "#bb931d", "#0B053A",
                    "#cf9202"]
    st.set_page_config(page_title="Kidney Disease EDA", layout="wide")
    st.title("Kidney Disease Prediction - Exploratory Data Analysis (EDA)")
    bins =[0,18,35,50,65,100]
    labels =['0-18','19-35','36-50','51-65','66+']
    data_analysis['Age_Group'] = pd.cut(data_analysis['Age'], bins=bins, labels=labels, right=False)
    st.markdown("### **This section provides analysis of CKD status in the dataset.**")
    col1,col2 = st.columns(2)
        
    with col1:
        st.markdown("**CKD Status by Age Group**")
        fig1, ax1 = plt.subplots(figsize=(14, 10))
        sns.countplot(data=data_analysis, x='Age_Group', hue='CKD_Status', palette=custom_palette[:2],ax=ax1)
        plt.title('CKD Status by Age Group')
        plt.xlabel('Age Group')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig1)
    with col2:
        st.markdown("**CKD Status by Diabetes Status**")
        fig2,ax2= plt.subplots(figsize=(14, 10))
        sns.countplot(data=data_analysis, x='Diabetes', hue='CKD_Status', palette=custom_palette[2:4],ax=ax2)
        plt.title('Diabetes Status by CKD Status')
        plt.xlabel('Diabetes Status')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='CKD Status')
        st.pyplot(fig2)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**CKD Status by Hypertension Status**")
        fig3, ax3 = plt.subplots(figsize=(14, 10))
        sns.countplot(data=data_analysis, x='Hypertension', hue='CKD_Status', palette=custom_palette[4:6], ax=ax3)
        plt.title('Hypertension Status by CKD Status')
        plt.xlabel('Hypertension Status')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='CKD Status')
        st.pyplot(fig3)

    with col4:
        st.markdown("**CKD Status by BUN Levels**")
        fig4, ax4 = plt.subplots(figsize=(14, 10))
        sns.boxplot(data=data_analysis,x='CKD_Status',y='BUN',palette=custom_palette[:2])
        plt.title('BUN Levels by CKD Status')
        plt.xlabel('CKD Status')
        plt.ylabel('BUN Levels')
        plt.xticks(rotation=45)
        st.pyplot(fig4)

    col5,col6 = st.columns(2)
    with col5:
        st.markdown("**CKD Status by Creatinine Levels**")
        fig5, ax5 = plt.subplots(figsize=(14, 10))
        sns.boxplot(data=data_analysis,x='CKD_Status',y='Creatinine_Level',palette=custom_palette[2:4], ax=ax5)
        plt.title('Creatinine Levels by CKD Status')
        plt.xlabel('CKD Status')
        plt.ylabel('Creatinine Levels')
        plt.xticks(rotation=45)
        st.pyplot(fig5)

    st.markdown("### **Dialysis Needed Analysis**")
    st.write("This section provides analysis of dialysis needed in the dataset.")    
    with col6:
        st.markdown("**CKD Status by Urine Output Levels**")
        fig6, ax6 = plt.subplots(figsize=(14, 10))
        sns.boxplot(data=data_analysis,x='CKD_Status',y='Urine_Output',palette=custom_palette[6:8], ax=ax6)
        plt.title('Urine Output Levels by CKD Status')
        plt.xlabel('CKD Status')
        plt.ylabel('Urine Output Levels')
        plt.xticks(rotation=45)
        st.pyplot(fig6)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Dialysis Needed by Age Group**")
        fig7, ax7 = plt.subplots(figsize=(14, 10))
        sns.countplot(data=data_analysis, x='Age_Group', hue='Dialysis_Needed', palette=custom_palette[6:8], ax=ax7)
        plt.title('Dialysis Needed by Age Group')
        plt.xlabel('Age Group')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        st.pyplot(fig7)

    with col2:
        st.markdown("**Dialysis Needed by Diabetes Status**")
        fig8, ax8 = plt.subplots(figsize=(14, 10))
        sns.countplot(data=data_analysis, x='Diabetes', hue='Dialysis_Needed', palette=custom_palette[2:4], ax=ax8)
        plt.title('Diabetes Status by Dialysis Needed')
        plt.xlabel('Diabetes Status')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='Dialysis Needed')
        st.pyplot(fig8)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Dialysis Needed by Hypertension Status**")
        fig9, ax9 = plt.subplots(figsize=(14, 10))
        sns.countplot(data=data_analysis, x='Hypertension', hue='Dialysis_Needed', palette=custom_palette[6:8], ax=ax9)
        plt.title('Hypertension Status by Dialysis Needed')
        plt.xlabel('Hypertension Status')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='Dialysis Needed')
        st.pyplot(fig9)

    with col4:
        st.markdown("**Dialysis Needed by BUN Levels**")
        fig10, ax10 = plt.subplots(figsize=(14, 10))
        sns.boxplot(data=data_analysis,x='Dialysis_Needed',y='BUN',palette=custom_palette[:2], ax=ax10)
        plt.title('BUN Levels by Dialysis Needed')
        plt.xlabel('Dialysis Needed')
        plt.ylabel('BUN Levels')
        plt.xticks(rotation=45)
        st.pyplot(fig10)

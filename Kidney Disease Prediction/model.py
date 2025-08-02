import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


def model_training(data_analysis):
    st.title("Model Training & Evaluation")
    st.write("This section allows you to train and evaluate machine learning models for kidney disease prediction.")
    
    if 'data_for_model' in st.session_state:
        model_data = st.session_state['data_for_model'].copy()
    else:
        st.info("Feature engineered data not found. Generating on the fly...")
        model_data = data_analysis.copy()

        # Reapply feature engineering
        model_data['BUN_Category'] = pd.cut(model_data['BUN'], bins=[0, 20, 40, 60, 80, 100], 
                                            labels=['Low', 'Medium', 'High', 'Very High', 'Extreme'], right=False)
        model_data['Creatinine_Category'] = pd.cut(model_data['Creatinine_Level'], bins=[0, 1, 2, 3, 4, 5], 
                                                labels=['Normal', 'Mild', 'Moderate', 'Severe', 'Critical'], right=False)
        model_data['GFR_category'] = pd.cut(model_data['GFR'], bins=[0, 15, 30, 45, 60, 75, 90, 120],
                                            labels=['Very Low', 'Low', 'Moderate', 'Mild', 'Normal', 'High', 'Very High'], right=False)
        model_data['Age_Group'] = pd.cut(model_data['Age'], bins=[0, 20, 40, 60, 80, 100], 
                                        labels=['0–20', '21–40', '41–60', '61–80', '81+'], right=False)

        # Drop unused columns
        model_data.drop(columns=['BUN', 'Creatinine_Level', 'Urine_Output', 'Dialysis_Needed', 'Age', 'GFR'], inplace=True)

    st.markdown("**1. Label Encoding**")
    le = LabelEncoder()
    for col in ['Diabetes', 'Hypertension', 'CKD_Status', 'BUN_Category', 'Creatinine_Category', 'Age_Group', 'GFR_category']:
        if col in model_data.columns:
            model_data[col] = le.fit_transform(model_data[col])

    st.write("Encoded data preview:")
    st.dataframe(model_data.head())

    st.markdown("**2. Data Splitting**")
    X = model_data.drop(columns=['CKD_Status'])
    y = model_data['CKD_Status']
    st.write("Features (X):", X.columns.tolist())
    st.write("Target (y): CKD_Status")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    with open("model_columns.pkl", "wb") as f:
        pickle.dump(X.columns.tolist(), f)

    st.markdown("**3. Model Training and Evaluation**")
    st.write("Training all available models: Logistic Regression, Decision Tree, and Random Forest")

    # Dictionary to store results for plotting
    model_results = {}
    classification_reports = {}
    trained_models = {}
    
    # Train all models and store results
    models = ["Logistic Regression", "Decision Tree", "Random Forest"]
    
    for model in models:
        if model == "Logistic Regression":
            lr_model = LogisticRegression()
            lr_model.fit(X_train, y_train)
            lr_preds = lr_model.predict(X_test)
            model_results[model] = {
                'predictions': lr_preds,
                'accuracy': accuracy_score(y_test, lr_preds),
                'confusion_matrix': confusion_matrix(y_test, lr_preds)
            }
            classification_reports[model] = classification_report(y_test, lr_preds, output_dict=True)
            trained_models[model] = lr_model
            
        elif model == "Decision Tree":
            dt_model = DecisionTreeClassifier()
            dt_model.fit(X_train, y_train)
            dt_preds = dt_model.predict(X_test)
            model_results[model] = {
                'predictions': dt_preds,
                'accuracy': accuracy_score(y_test, dt_preds),
                'confusion_matrix': confusion_matrix(y_test, dt_preds)
            }
            classification_reports[model] = classification_report(y_test, dt_preds, output_dict=True)
            trained_models[model] = dt_model
            
        elif model == "Random Forest":
            rf_model = RandomForestClassifier()
            rf_model.fit(X_train, y_train)
            rf_preds = rf_model.predict(X_test)
            model_results[model] = {
                'predictions': rf_preds,
                'accuracy': accuracy_score(y_test, rf_preds),
                'confusion_matrix': confusion_matrix(y_test, rf_preds)
            }
            classification_reports[model] = classification_report(y_test, rf_preds, output_dict=True)
            trained_models[model] = rf_model

    # Display accuracy scores
    st.subheader("Model Accuracy Comparison")
    accuracy_df = pd.DataFrame([
        {"Model": model, "Accuracy": f"{results['accuracy']:.4f}"} 
        for model, results in model_results.items()
    ])
    st.dataframe(accuracy_df, use_container_width=True)

    # Display confusion matrices in columns
    st.subheader("Confusion Matrices")
    cols = st.columns(3)
    for i, (model_name, results) in enumerate(model_results.items()):
        with cols[i]:
            cm = results['confusion_matrix']
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Not CKD', 'CKD'], yticklabels=['Not CKD', 'CKD'], ax=ax)
            ax.set_title(f'{model_name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            plt.close()

    # Display classification reports as markdown tables
    st.subheader("Classification Reports")
    for model_name, report in classification_reports.items():
        st.markdown(f"**{model_name} Classification Report**")
        
        # Convert classification report to markdown table
        report_data = []
        for class_name, metrics in report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                class_label = 'Not CKD' if class_name == '0' else 'CKD' if class_name == '1' else class_name
                report_data.append({
                    'Class': class_label,
                    'Precision': f"{metrics['precision']:.3f}",
                    'Recall': f"{metrics['recall']:.3f}",
                    'F1-Score': f"{metrics['f1-score']:.3f}",
                    'Support': int(metrics['support'])
                })
        
        # Add summary rows
        if 'macro avg' in report:
            report_data.append({
                'Class': 'Macro Avg',
                'Precision': f"{report['macro avg']['precision']:.3f}",
                'Recall': f"{report['macro avg']['recall']:.3f}",
                'F1-Score': f"{report['macro avg']['f1-score']:.3f}",
                'Support': int(report['macro avg']['support'])
            })
        
        if 'weighted avg' in report:
            report_data.append({
                'Class': 'Weighted Avg',
                'Precision': f"{report['weighted avg']['precision']:.3f}",
                'Recall': f"{report['weighted avg']['recall']:.3f}",
                'F1-Score': f"{report['weighted avg']['f1-score']:.3f}",
                'Support': int(report['weighted avg']['support'])
            })
        
        # Display as markdown table
        report_df = pd.DataFrame(report_data)
        st.markdown(report_df.to_markdown(index=False))
        st.markdown("---")  # Separator between models

    # Find and save the best model
    best_model_name = max(model_results, key=lambda name: model_results[name]['accuracy'])
    best_model = trained_models[best_model_name]
    
    st.subheader("Best Model Selection")
    st.success(f"Best performing model: **{best_model_name}** with accuracy: {model_results[best_model_name]['accuracy']:.4f}")

    # Save the best model
    try:
        with open(f"{best_model_name.replace(' ', '_').lower()}_model.pkl", 'wb') as f:
            pickle.dump(best_model, f)
        st.success(f"{best_model_name} saved successfully!")
    except Exception as e:
        st.error(f"Error saving model: {e}")

    # Save encoders for consistent prediction preprocessing
    try:
        encoders = {}
        for col in ['Diabetes', 'Hypertension', 'CKD_Status', 'BUN_Category', 'Creatinine_Category', 'Age_Group', 'GFR_category']:
            if col in model_data.columns:
                le = LabelEncoder()
                # Fit on the original data to get all possible categories
                le.fit(model_data[col])
                encoders[col] = le

        # Save encoders
        with open("label_encoders.pkl", "wb") as f:
            pickle.dump(encoders, f)
        st.success("Label encoders saved successfully!")
    except Exception as e:
        st.error(f"Error saving encoders: {e}")

    # Save scaler
    try:
        with open("scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        st.success("Scaler saved successfully!")
    except Exception as e:
        st.error(f"Error saving scaler: {e}")
import streamlit as st
import pandas as pd
import pickle
import numpy as np

def model_prediction():
    st.title("Model Prediction")
    st.write("This section allows you to make predictions using the trained model.")
    
    # Load model, encoders, and scaler
    try:
        with open("random_forest_model.pkl", 'rb') as f:
            model = pickle.load(f)
        with open("label_encoders.pkl", 'rb') as f:
            encoders = pickle.load(f)
        with open("scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        st.success("Model, encoders, and scaler loaded successfully!")
        with open("model_columns.pkl", "rb") as f:
            model_columns = pickle.load(f)



    except FileNotFoundError as e:
        st.error(f"File not found: {str(e)}. Please train and save the model first.")
        st.stop()
    
    age_bins = [0, 20, 40, 60, 80, 100]
    age_labels = ['0–20', '21–40', '41–60', '61–80', '81+']
    age_numeric = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
    
    # Create age category manually to match original encoding
    if age_numeric < 20:
        age_category_auto = '0–20'
    elif age_numeric < 40:
        age_category_auto = '21–40'
    elif age_numeric < 60:
        age_category_auto = '41–60'
    elif age_numeric < 80:
        age_category_auto = '61–80'
    else:
        age_category_auto = '81+'
    
    age_category = st.selectbox("Age Group", age_labels, index=age_labels.index(age_category_auto))
    
    # BUN
    bun_categories = ['Low', 'Medium', 'High', 'Very High', 'Extreme']
    bun_numeric = st.number_input("BUN Level", min_value=0.0, max_value=200.0, value=20.0, step=1.0)
    
    # Create BUN category with proper bins
    if bun_numeric < 20:
        bun_category_auto = 'Low'
    elif bun_numeric < 40:
        bun_category_auto = 'Medium'
    elif bun_numeric < 60:
        bun_category_auto = 'High'
    elif bun_numeric < 80:
        bun_category_auto = 'Very High'
    else:
        bun_category_auto = 'Extreme'
    
    bun_category = st.selectbox("BUN Category", bun_categories, 
                               index=bun_categories.index(bun_category_auto))
    
    # Creatinine
    creatinine_bins = [0, 1, 2, 3, 4, 5]
    creatinine_labels = ['Normal', 'Mild', 'Moderate', 'Severe', 'Critical']
    creatinine_level = st.number_input("Creatinine Level", min_value=0.0, max_value=10.0, 
                                     value=1.0, step=0.1, format="%.2f")
    
    # Create creatinine category manually to avoid NaN issues
    if creatinine_level < 1:
        creatinine_category_auto = 'Normal'
    elif creatinine_level < 2:
        creatinine_category_auto = 'Mild'
    elif creatinine_level < 3:
        creatinine_category_auto = 'Moderate'
    elif creatinine_level < 4:
        creatinine_category_auto = 'Severe'
    else:
        creatinine_category_auto = 'Critical'
    
    creatinine_category = st.selectbox("Creatinine Category", creatinine_labels, 
                                     index=creatinine_labels.index(creatinine_category_auto))
    
    # GFR - Using original encoding labels
    gfr_bins = [0, 15, 30, 45, 60, 75, 90, 120]
    gfr_labels = ['Very Low', 'Low', 'Moderate', 'Mild', 'Normal', 'High', 'Very High']
    gfr_level = st.number_input("GFR Level", min_value=0.0, max_value=150.0, value=60.0, step=1.0)
    
    # Create GFR category manually to match original encoding
    if gfr_level < 15:
        gfr_category_auto = 'Very Low'
    elif gfr_level < 30:
        gfr_category_auto = 'Low'
    elif gfr_level < 45:
        gfr_category_auto = 'Moderate'
    elif gfr_level < 60:
        gfr_category_auto = 'Mild'
    elif gfr_level < 75:
        gfr_category_auto = 'Normal'
    elif gfr_level < 90:
        gfr_category_auto = 'High'
    else:
        gfr_category_auto = 'Very High'
    
    gfr_category = st.selectbox("GFR Category", gfr_labels, 
                               index=gfr_labels.index(gfr_category_auto))
    
    # Diabetes & Hypertension
    diabetes = st.radio("Diabetes Status", ["No", "Yes"])
    hypertension = st.radio("Hypertension Status", ["No", "Yes"])
 
    input_data = pd.DataFrame({
        'Diabetes': [diabetes],
        'Hypertension': [hypertension], 
        'BUN_Category': [bun_category],
        'Creatinine_Category': [creatinine_category],
        'GFR_category': [gfr_category],
        'Age_Group': [age_category]
    })
    
    
    # Manual encoding based on training data categories
    encoding_maps = {
        'Age_Group': {'0–20': 0, '21–40': 1, '41–60': 2, '61–80': 3, '81+': 4},
        'BUN_Category': {'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3, 'Extreme': 4},
        'Creatinine_Category': {'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3, 'Critical': 4},
        'GFR_category': {'Very Low': 0, 'Low': 1, 'Moderate': 2, 'Mild': 3, 'Normal': 4, 'High': 5, 'Very High': 6},
        'Diabetes': {'No': 0, 'Yes': 1},
        'Hypertension': {'No': 0, 'Yes': 1}
    }
    
    # Apply manual encoding
    for col in input_data.columns:
        if col in encoding_maps:
            value = input_data[col].iloc[0]
            if value in encoding_maps[col]:
                input_data[col] = encoding_maps[col][value]
            else:
                st.error(f"Unknown value '{value}' for column '{col}'")
                st.write(f"Expected values: {list(encoding_maps[col].keys())}")
                st.stop()

    if st.button("Predict"):
        try:
            # Ensure correct column order
            input_data = input_data[model_columns]
            
            # Scale the input data
            input_data_scaled = scaler.transform(input_data)
            input_data_scaled = pd.DataFrame(input_data_scaled, columns=model_columns)
            
            # Make prediction
            prediction = model.predict(input_data_scaled)
            prediction_proba = model.predict_proba(input_data_scaled)
            
            prediction_label = "CKD" if prediction[0] == 1 else "Not CKD"
            confidence = max(prediction_proba[0]) * 100
            
            st.success(f"Prediction: {prediction_label}")
            st.info(f"Confidence: {confidence:.2f}%")
            
            # Show probability for both classes
            st.write("Prediction Probabilities:")
            prob_df = pd.DataFrame({
                'Class': ['Not CKD', 'CKD'],
                'Probability': [prediction_proba[0][0], prediction_proba[0][1]]
            })
            
            st.bar_chart(prob_df.set_index('Class'))

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

import streamlit as st
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from streamlit_option_menu import option_menu
from sklearn.preprocessing import LabelEncoder

parkinson_model = joblib.load('parkinson_model.pkl')
kidney_model = joblib.load('kidney_model.pkl')
liver_model = joblib.load('liver_model.pkl')


#Handle the outliers in the dataset
def handle_outliers_iqr(df):
    numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
    outliers = []
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Element-wise comparison to find outliers
        lower_outliers = df[col] < lower_bound
        upper_outliers = df[col] > upper_bound
        
        if lower_outliers.any() or upper_outliers.any(): 
            # Check if any outliers exist
            outliers.append(df[col].name)
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
            #st.write("outliers filled")
    return df,outliers  

def fill_missing_values(df):
    for col in df.columns:
        if df[col].isnull().sum() > 0:  # If there are missing values
            if df[col].dtype in ['int64', 'float64']:  # For numeric columns
                df[col].fillna(df[col].mean(), inplace=True)
            else:  # For object (categorical) columns
                df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def encode_categorical_values(df):
    label_encoder = LabelEncoder()
    for column in df.columns:
        # Check if the column is of object dtype (typically for categorical/string data)
        if df[column].dtype == 'object':
            # Apply label encoding to the column
            df[column] = label_encoder.fit_transform(df[column])
    return df

#Horizontal Navigation menu
with  st.sidebar:
    selected = option_menu(
        menu_title = "",
        options = ["About Project","Parkinsons disease prediction","Kidney disease prediction",
               "Indian liver patient prediction"],
        icons = ["info-square-fill","info-square-fill","info-square-fill","info-square-fill"],
        default_index=0)

if selected == "About Project":
    st.snow()
    st.title("Multiple Disease Prediction")
    st.markdown(""" 
    ##### Objective:
    **To build a scalable and accurate system that assists in early detection of diseases,improves decision-making for healthcare providers and
    reduces diagnostic time and cost by providing quick predictions.**
    ##### Skills take away from this project : 
    **Streamlit, Python,Machine Learning, Visualization.**   
    """)

elif selected == "Parkinsons disease prediction":
    st.write("#### Parkinsons disease prediction")
    col1, col2, col3 = st.columns(3)
    with col1:
        MDVP_Fo_Hz = st.number_input("MDVP_Fo(Hz)",value=119.992,format="%.7f")
        MDVP_Jitter = st.number_input("MDVP_Jitter(%)",value=0.00784,format="%.7f")
        PPQ = st.number_input("MDVP_PPQ",value=0.00554,format="%.7f")
        MDVP_Shimmer_dB = st.number_input("MDVP_Shimmer(dB)",value=0.426,format="%.7f")
        MDVP_APQ = st.number_input("MDVP_APQ",value=0.02971,format="%.7f")
        HNR = st.number_input("HNR",value=21.033,format="%.7f")
        spread1 = st.number_input("spread1",value=-4.813031,format="%.7f")
        PPE = st.number_input("PPE",value=0.284654,format="%.7f")

    with col2:
        MDVP_Fhi_Hz = st.number_input("MDVP_Fhi(Hz)",value=157.302,format="%.7f")
        MDVP_Jitter_Abs = st.number_input("MDVP_Jitter(Abs)",value=0.00007,format="%.7f")
        DDP = st.number_input("Jitter_DDP",value=0.01109,format="%.7f")
        Shimmer_APQ3 = st.number_input("Shimmer_APQ3",value=0.02182,format="%.7f")
        Shimmer_DDA = st.number_input("Shimmer_DDA",value=0.06545,format="%.7f")
        RPDE = st.number_input("RPDE",value=0.414783,format="%.7f")
        spread2 = st.number_input("spread2",value=0.266482,format="%.7f")
    with col3:
        MDVP_Flo_Hz = st.number_input("MDVP_Flo(Hz)",value=74.997,format="%.7f")
        RAP = st.number_input("MDVP_RAP",value=0.0037,format="%.7f")
        MDVP_Shimmer = st.number_input("MDVP_Shimmer",value=0.04374,format="%.7f")
        Shimmer_APQ5 = st.number_input("Shimmer_APQ5",value=0.0313,format="%.7f")
        NHR = st.number_input("NHR",value=0.02211,format="%.7f")
        DFA = st.number_input("DFA",value=0.815285,format="%.7f")
        D2 = st.number_input("D2",value=2.301442,format="%.7f")
    data=np.array([[MDVP_Fo_Hz,MDVP_Fhi_Hz,MDVP_Flo_Hz,MDVP_Jitter,MDVP_Jitter_Abs,RAP,PPQ,DDP,MDVP_Shimmer,
          MDVP_Shimmer_dB,Shimmer_APQ3,Shimmer_APQ5,MDVP_APQ,Shimmer_DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])
    
    if data.all and st.button("Predict"):
        prediction = parkinson_model.predict(data)
        if prediction[0] == 0:
            st.success("The Person does not have a parkinson's disease")
        else:
            st.error("The Person has parkinson's disease")

elif selected == "Kidney disease prediction":
    st.write("#### Kidney disease prediction")
    col1, col2, col3 = st.columns(3)
    with col1:
        age  = st.number_input("Age",value=58)
        sugar = st.number_input("Sugar(su)",value=0)
        pus_cell_clumps = st.number_input("Pus Cell Clumps(pcc)",value=0)
        serum_creatinine = st.number_input("Serum Creatinine(sc)",value=1.2)
        red_blood_cell_count = st.number_input("Red Blood Cell Count (rc)",value=4.9)
        coronary_artery_disease = st.number_input("Coronary Artery Disease (cad)",value=0)
    with col2:
        blood_pressure  = st.number_input("Blood Pressure(bp)",value=70)
        red_blood_cells = st.number_input("Red Blood Cells(rbc)",value=1)
        blood_glucose_random = st.number_input("Blood Glucose Random(bgr)",value=102)
        hemoglobin  = st.number_input("Hemoglobin(hemo)",value=15.2)
        hypertension = st.number_input("Hypertension(htn)",value=0)
        pedal_edema = st.number_input("Pedal Edema(pe)",value=0)
    with col3:
        albumin = st.number_input("Albumin(al)",value=0)
        pus_cells = st.number_input("Pus Cells(pc)",value=1)
        blood_urea = st.number_input("Blood Urea(bu)",value=48)
        white_blood_cells = st.number_input("White Blood Cells (wc)",value=8100)
        diabetes  = st.number_input("Diabetes(dm) ",value=0)
        anemia  = st.number_input("Anemia",value=0)
    data=np.array([[age,blood_pressure,albumin,sugar,red_blood_cells,pus_cells,pus_cell_clumps,blood_glucose_random,blood_urea,
                serum_creatinine,hemoglobin,white_blood_cells,red_blood_cell_count,hypertension,diabetes,
                coronary_artery_disease,pedal_edema,anemia]])
    if data.all and st.button("Predict"):
        prediction = kidney_model.predict(data)
        if prediction[0] == 0:
            st.error("The Person has chronic kidney disease")
        else:
            st.success("The Person does not have a chronic kidney disease")
    
elif selected == "Indian liver patient prediction":
        st.write("#### Kidney disease prediction")
        col1, col2, col3 = st.columns(3)
        with col1:
            age  = st.number_input("Age",value=65)
            alkaline_phosphotase  = st.number_input("Alkaline_Phosphotase",value=187)
            total_protiens  = st.number_input("Total_Protiens",value=6.8)
        with col2:
            total_bilirubin  = st.number_input("Total_Bilirubin",value=0.7)
            alamine_aminotransferase  = st.number_input("Alamine_Aminotransferase",value=16)
            albumin  = st.number_input("Albumin",value=3.3)
        with col3:
            direct_bilirubin  = st.number_input("Direct_Bilirubin",value=0.1)
            aspartate_aminotransferase  = st.number_input("Aspartate_Aminotransferase",value=18)
            albumin_and_globulin_ratio  = st.number_input("Albumin_and_Globulin_Ratio",value=0.9)

        data=np.array([[age,total_bilirubin,direct_bilirubin,alkaline_phosphotase,alamine_aminotransferase,aspartate_aminotransferase,
                        total_protiens, albumin,albumin_and_globulin_ratio]])

        if data.all and st.button("Predict"):
            prediction = liver_model.predict(data)
            if prediction[0] == 1:
                st.error("The Person has Liver disease")
            else:
                st.success("The Person does not have a Liver disease")

    











    













import pandas as pd
import joblib
import multiple_disease_prediction as mdp
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import  precision_score,f1_score,recall_score
import streamlit as st


liver_df = pd.read_csv(r"E:\User\Desktop\sham\DATA SCIENCE\PROJECTS\Project3 - Multiple Disease Prediction\files\indian_liver_patient - indian_liver_patient.csv")
# Drop unwanted columns
df = liver_df.drop(['Gender'], axis=1)

#Data preprocessing
#Fill the outliers using capping method
df,outliers = mdp.handle_outliers_iqr(df)
#Fill missing values if any
df = mdp.fill_missing_values(df)

target = df['Dataset']
feature = df.drop(['Dataset'],axis=1)

# Split the data into training and testing sets
feature_train, feature_test, target_train, target_test = train_test_split(feature, target, test_size=0.2, random_state=42)
    
#Train the model
liver_model = DecisionTreeClassifier()
liver_model.fit(feature_train,target_train)
predicted_value = liver_model.predict(feature_test)

#Evaluate the model
precision = precision_score(target_test,predicted_value)
recall = recall_score(target_test,predicted_value)
score = f1_score(target_test,predicted_value)

#save the model using joblib
st.write("joblib")
joblib.dump(liver_model,'liver_model.pkl')
import pandas as pd
import joblib
import streamlit as st
import multiple_disease_prediction as mdp
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import  precision_score,f1_score,recall_score


kidney_df = pd.read_csv(r"E:\User\Desktop\sham\DATA SCIENCE\PROJECTS\Project3 - Multiple Disease Prediction\files\kidney_disease - kidney_disease.csv")
# Drop unwanted columns
df = kidney_df.drop(['id','sg','ba','sod','pot','pcv','appet'], axis=1)

#Data preprocessing
#Fill the outliers using capping method
df,outliers = mdp.handle_outliers_iqr(df)
#Fill missing values if any
df = mdp.fill_missing_values(df)

#Encode categorical values
df = mdp.encode_categorical_values(df)

target = df['classification']
feature = df.drop(['classification'],axis=1)

# Split the data into training and testing sets
feature_train, feature_test, target_train, target_test = train_test_split(feature, target, test_size=0.2, random_state=42)
    
#Train the model
kidney_model = DecisionTreeClassifier()
kidney_model.fit(feature_train,target_train)
predicted_value = kidney_model.predict(feature_test)

#Evaluate the model
precision = precision_score(target_test,predicted_value)
recall = recall_score(target_test,predicted_value)
score = f1_score(target_test,predicted_value)

#save the model using joblib
joblib.dump(kidney_model,'kidney_model.pkl')
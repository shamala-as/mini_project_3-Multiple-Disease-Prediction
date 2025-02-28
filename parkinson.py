import pandas as pd
import joblib
import streamlit as st
import multiple_disease_prediction as mdp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  precision_score,f1_score,recall_score


parkinsons_df = pd.read_csv(r"E:\User\Desktop\sham\DATA SCIENCE\PROJECTS\Project3 - Multiple Disease Prediction\files\parkinsons - parkinsons.csv")

#Data preprocessing

#Fill the outliers using capping method
df,outliers = mdp.handle_outliers_iqr(parkinsons_df.drop(columns=['name','status','MDVP:RAP','Jitter:DDP','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','Shimmer:DDA']))
#Fill missing values if any
df = mdp.fill_missing_values(df)

feature = df
target = parkinsons_df['status']
# Split the data into training and testing sets
feature_train, feature_test, target_train, target_test = train_test_split(feature, target, test_size=0.2, random_state=42)

#Train the model
parkinson_model = LogisticRegression()
parkinson_model.fit(feature_train,target_train)
predicted_value = parkinson_model.predict(feature_test)

#Evaluate the model
precision = precision_score(target_test,predicted_value)
recall = recall_score(target_test,predicted_value)
score = f1_score(target_test,predicted_value)

#save the model using joblib
joblib.dump(parkinson_model,'parkinson_model.pkl')
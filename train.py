import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder ,OneHotEncoder

from sklearn.metrics import accuracy_score, classification_report
# load the dataset
df=pd.read_csv('data/Churn_Modelling.csv')  # Load your dataset

print(df.head())  # Display the first few rows of the dataset

#Label encoding for 'Gender' column
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])

#One hot encoding for 'Geography' column
one_hot_encoded =OneHotEncoder(sparse=False)
df_encoded = pd.DataFrame(one_hot_encoded.fit_transform(df[['Geography']]), columns=one_hot_encoded.get_feature_names_out(['Geography']))

print(df_encoded.head())  # Display the first few rows of the encoded dataset

#feature selection
X = df.drop(['Exited', 'RowNumber', 'Surname'], axis=1)
y = df['Exited']

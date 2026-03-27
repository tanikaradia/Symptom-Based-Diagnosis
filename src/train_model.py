import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import joblib

# print(os.getcwd()) #Current working directory

# Paths setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_path = os.path.join(BASE_DIR, 'data', 'training.csv')
test_path = os.path.join(BASE_DIR, 'data', 'testing.csv')
# print("File path:", file_path)

# Load dataset
train_df = pd.read_csv(train_path, encoding='latin1')
test_df = pd.read_csv(test_path, encoding='latin1')
# df = pd.read_csv(file_path, encoding='latin1') #messy-char to readable

# Data cleaning
train_df = train_df.drop_duplicates().dropna()
test_df = test_df.drop_duplicates().dropna()

print("After cleaning:")
print(train_df.info())
# print(df.head())

# Features=x,symptoms & Target=y,disease/prognosis split
# training data
X_train = train_df.drop('Prognosis', axis=1)
y_train = train_df['Prognosis']

# testing data
X_test = test_df.drop('Prognosis', axis=1)
y_test = test_df['Prognosis']

# Feature filtering
# X_train = X_train.loc[:, (X_train != 0).any(axis=0)] #select those columns,T/F,check column-wise
# X_test = X_test[X_train.columns] #test data =same columns= train data

print(X_train.columns.equals(X_test.columns))

# Shape check
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# Model creation
model = RandomForestClassifier(class_weight='balanced') #performance up of minority class, not biased

# Train Model
model.fit(X_train, y_train) #learnt patterns symptoms → disease mapping

print("Model trained ✅")

# Evaluate Model
# Prediction
y_pred = model.predict(X_test)

probs = model.predict_proba(X_test)
print("Confidence (first prediction):", max(probs[0])) #max->confidence

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
model_path = os.path.join(BASE_DIR, 'models', 'model.pkl')
joblib.dump(model, model_path)

print("Model saved ✅")
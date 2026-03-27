import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# print(os.getcwd()) #Current working directory

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_path = os.path.join(BASE_DIR, 'data', 'training.csv') #for project_folder/data/trainings.csv
test_path = os.path.join(BASE_DIR, 'data', 'testing.csv')

# print("File path:", file_path)

# load dataset
train_df = pd.read_csv(train_path, encoding='latin1')
test_df = pd.read_csv(test_path, encoding='latin1')
# df = pd.read_csv(file_path, encoding='latin1') #messy-char to readable

train_df = train_df.drop_duplicates().dropna()
test_df = test_df.drop_duplicates().dropna()

print("After cleaning:")
print(train_df.info())
# print(df.head())

# # features (symptoms) x
# # target (disease/prognosis) y
# training data
X_train = train_df.drop('Prognosis', axis=1)
y_train = train_df['Prognosis']

# testing data
X_test = test_df.drop('Prognosis', axis=1)
y_test = test_df['Prognosis']

# X_train = X_train.loc[:, (X_train != 0).any(axis=0)]
# X_test = X_test[X_train.columns] 

print(X_train.columns.equals(X_test.columns))

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

model = RandomForestClassifier(class_weight='balanced')
# model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train Model
model.fit(X_train, y_train)

print("Model trained ✅")

# Evaluate Model
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
import pandas as pd
import os
from sklearn.model_selection import train_test_split
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

# print(X_train.columns.equals(X_test.columns))

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
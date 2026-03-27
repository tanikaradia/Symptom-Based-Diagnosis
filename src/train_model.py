import pandas as pd
import os
# print(os.getcwd()) #Current working directory

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #2folderout,fullpath,of curr =root folder
file_path = os.path.join(BASE_DIR, 'data', 'trainings.csv') #for project_folder/data/trainings.csv

# print("File path:", file_path)

# load dataset
df = pd.read_csv(file_path, encoding='latin1') #messy-char to readable

# remove duplicate rows
df = df.drop_duplicates()

# remove missing values
df = df.dropna()

print("After cleaning:")
print(df.info()) #sz
# print(df.head())

# features (symptoms)
X = df.drop('Prognosis', axis=1)

# target (disease/prognosis)
y = df['Prognosis']

print("X shape:", X.shape)  #drop col(y=1) named prognosis,2D
print("y shape:", y.shape)  #2D, 391row & 1col

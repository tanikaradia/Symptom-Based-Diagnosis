import pandas as pd
import os
# print(os.getcwd()) #Current working directory

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #2folderout,fullpath,of curr =root folder
file_path = os.path.join(BASE_DIR, 'data', 'trainings.csv') #for project_folder/data/trainings.csv

# print("File path:", file_path)

# load dataset
df = pd.read_csv(file_path, encoding='latin1') #messy-char to readable
# df = pd.read_csv('../data/trainings.csv')

print(df.head())
print(df.info()) #sz
#File path dynamically banaya,CSV load kiya,Data preview + structure check kiya


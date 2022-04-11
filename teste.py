import pandas as pd


with open('/home/thales/KT-Models/AKT/data/errex/errex_replace_data.csv','rb') as file:
    df = pd.read_csv(file)


print(len(df['student_id'].unique()))
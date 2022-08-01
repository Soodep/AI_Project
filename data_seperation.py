
import pandas as pd

df = pd.read_csv("train.csv", encoding = "ISO-8859-1")

df_y = df[['label']]

df_x = df[['feature']]

df_y.to_csv('label.csv', index=False, header=False)

df_x.to_csv('data.csv', index=False, header=False)


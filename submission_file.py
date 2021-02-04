import pandas as pd
import numpy as np

df = pd.read_csv('competitive-data-science-predict-future-sales/submissions/sample_submission.csv')
print(df.head())
new_constant = 0.35
current_constant = df.item_cnt_month.mean()

n_df = df.replace(current_constant, new_constant)
print(n_df.head())
n_df.to_csv('./competitive-data-science-predict-future-sales/new_submission.csv', index=False)

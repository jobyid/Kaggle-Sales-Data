import pandas as pd
import numpy as np

df = pd.read_csv('./competitive-data-science-predict-future-sales/sales_train.csv')
df = df.fillna(0)
df_oct = df[df.date_block_num == 33]
oct15 = df_oct.groupby(by=["item_id","shop_id"])['item_cnt_day'].agg(['sum']).reset_index()
te_df = pd.read_csv('./competitive-data-science-predict-future-sales/test.csv')
pred_df = pd.merge(te_df, oct15, how='left', on=["shop_id", "item_id"])
sub = pred_df[['ID','sum']]
sub = sub.fillna(0)

sub = sub['sum'].clip(lower=0, upper=20)
subs = pd.read_csv('competitive-data-science-predict-future-sales/submissions/sample_submission.csv')
subs.item_cnt_month = sub.sum
subs.to_csv('./competitive-data-science-predict-future-sales/last_month_submission.csv', index=False)
print(sub.shape)
print(subs.shape)

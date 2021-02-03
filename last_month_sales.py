import pandas as pd
import numpy as np

df = pd.read_csv('./competitive-data-science-predict-future-sales/sales_train.csv')
df = df.fillna(0)
#print(df.info())
df_oct = df[df.date_block_num == 33]
#print(df_oct.info())
shop_items = df_oct.groupby(by=["item_id","shop_id"])

month = pd.DataFrame({'shop_id':[],'item_id':[],'item_cnt_month':[]})
for s, i in shop_items:
    sh = s[1]
    it = s[0]
    sa = i.item_cnt_day.sum()
    month = month.append({'shop_id':sh,'item_id':it,'item_cnt_month':sa}, ignore_index=True)

month.item_cnt_month = month.item_cnt_month.clip(lower=0, upper=20)

te_df = pd.read_csv('./competitive-data-science-predict-future-sales/test.csv')

pred_df = pd.merge(te_df, month, how='left')
print("test shape ", te_df.shape )
print("pred shape ", pred_df.columns)
sub = pred_df[['ID','item_cnt_month']]
sub = sub.fillna(0)
sub.to_csv('./competitive-data-science-predict-future-sales/last_month_submission.csv', index=False)
print(sub.shape)

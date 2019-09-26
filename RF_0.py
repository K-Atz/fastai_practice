from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from fastai.imports import *
from fastai.structured import *

from sklearn import metrics

PATH = "../kaggle/"
df_raw = pd.read_csv(f'{PATH}Train.csv', low_memory=False, 
                     parse_dates=["saledate"])

df_raw.SalePrice = np.log(df_raw.SalePrice)
add_datepart(df_raw, 'saledate')
train_cats(df_raw)
df_raw.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)
df_raw.UsageBand = df_raw.UsageBand.cat.codes
df, y, nas = proc_df(df_raw, 'SalePrice')

m = RandomForestRegressor(n_jobs=-1)
m.fit(df, y)
print("score: ", m.score(df,y))
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from fastai.imports import *
from fastai.structured import *
from sklearn import metrics

def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)

df_raw = pd.read_feather('feathers/bulldozers-raw-beforetraincats')
df, y, nas = proc_df(df_raw, 'SalePrice')
# print(nas) not sure what is this
m = RandomForestRegressor(n_jobs=-1)
m.fit(df, y)
print("score: ", m.score(df,y))
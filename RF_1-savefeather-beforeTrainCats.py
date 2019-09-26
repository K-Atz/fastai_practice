from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from fastai.imports import *
from fastai.structured import *
from sklearn import metrics

def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)

PATH = "../kaggle/"
df_raw = pd.read_csv(f'{PATH}Train.csv', low_memory=False, 
                     parse_dates=["saledate"])

df_raw.SalePrice = np.log(df_raw.SalePrice)
add_datepart(df_raw, 'saledate')

df_raw.to_feather('feathers/bulldozers-raw-beforetraincats')
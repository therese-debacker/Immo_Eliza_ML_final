import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('./data/merged-data.csv')

# OUTLIERS
df.drop(df[df['Price'] > 2500000].index, inplace = True)
df.drop(df[df['Property type'] == 'Other_Property'].index, inplace = True)
df.drop(df[(df['Property type'] == 'Mixed_Use_Building') & (df['Living area'] > 1200)].index, inplace = True)
df.drop(df[(df['Property'] == 'Apartment') & (df['Living area'] > 450)].index, inplace = True)
df.drop(df[(df['Property'] == 'Apartment') & (df['Price'] > 1000000)].index, inplace = True)
df.drop(df[(df['Property'] == 'House') & (df['Living area'] > 1200)].index, inplace = True)
df.drop(df[(df['Property'] != 'House') & (df['Property'] != 'Apartment')].index, inplace = True)


# Removings rows
def remove_rows(df, column_name):
    df.dropna(subset=[column_name], inplace=True)

remove_rows(df, 'Living area')

# Too many missing values in surface of the plot --> replacing by the median (not mean because of outliers)
def replace_navalues(df, column_name, group):
    df[column_name] = df.groupby(group)[column_name].transform(lambda x: x.fillna(x.median()))

replace_navalues(df, 'Surface of the plot', 'district')


def replace_navalues_groups(df, column_name, group1, group2):
    df[column_name] = df.groupby([group1, group2])[column_name].transform(lambda x: x.fillna(x.median()))
replace_navalues_groups(df, 'median-price', 'Province', 'Property')

df.drop(df[df['Building condition'] == 'Not mentioned'].index, inplace = True)
df['Building condition'] = df['Building condition'].replace({'As new':6,'Just renovated':5,'Good':4,'To be done up':3,'To renovate':2,'To restore':1})


# test
columns_to_drop = ['Facades','Equipped kitchen', 'Furnished', 'Fireplace','Garden', 'Terrace', 'Terrace surface','Region', 'Bedrooms',
                    'Zip code', 'Locality', 'median-income', 'population', 'commune','locality', 'Province','Garden surface',
                    'poverty-chance', 'Property', 'surface-area-total','surface-area-total-built','number-parcels-land', 'surface-area-total-land',
                    'nb_transactions_house', 'nb_transactions_apartment','mean-income-district', 'median-income-district', 'poverty-chance-district',
                    'population-district', 'house-median-price', 'apartment-median-price', 'Unnamed: 0', 'number-parcels-built', 
                    'population-per-surface-district', 'number-transactions' ]

def removing_columns(columns_to_drop):
    df.drop(columns=columns_to_drop, inplace=True)

removing_columns(columns_to_drop)

print(df.columns)

# Transform categorical columns into 1/0
df = pd.get_dummies(df, columns=['Property type', 'district'], drop_first=True)


df.to_csv('./data/dataset-preprocessed.csv')
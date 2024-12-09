# Import necessary libraries and dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('./data/merged-data.csv')

# Getting a first look at the dataset to choose the features will use in our model and know what changes we have to make
def dataset_check_graphs_info(part: str):
    """
    Function that will give us an overview of the dataframe and create a correlation map (a general one and one per house and per apartment)
    :Parameter: a name to include in the heatmap name to be able to create the heatmap several times without erasing the previous ones
    """
    print(df.columns)
    print(df.describe())
    print(df.info())
    print(df.dtypes)
    print(df.isnull().sum())
    # Heatmaps
    plt.figure(figsize=(30, 20))
    sns.heatmap(df.corr(numeric_only=True), annot= True)
    plt.tight_layout()
    plt.savefig(f"./graphs/correlation-heatmap-{part}.png")
    plt.clf()
    df_houses = df.drop(df[df['Property'] != 'House'].index)
    sns.heatmap(df_houses.corr(numeric_only=True), annot= True)
    plt.tight_layout()
    plt.savefig(f"./graphs/correlation-heatmap-houses-{part}.png")
    plt.clf()
    df_apartments = df.drop(df[df['Property'] != 'Apartment'].index)
    sns.heatmap(df_apartments.corr(numeric_only=True), annot= True)
    plt.tight_layout()
    plt.savefig(f"./graphs/correlation-heatmap-apartment-{part}.png")
    plt.clf()
    # Boxplot of the price
    sns.boxplot(data=df.Price)
    step = 500000
    max_price = df['Price'].max() 
    yticks = np.arange(0, max_price + step, step)
    plt.yticks(yticks, labels=[f"{int(y):,}" for y in yticks]) 
    plt.grid(alpha=0.5, linestyle='--')
    plt.savefig(f"./graphs/boxplot-price.png")

dataset_check_graphs_info("one")

# Creating graphs to have a better view of the dataset
def check_closer_corr(column: str):
    """
    Function that creates a scatterplot based on one column and the price and adding colours based on the property type
    :Parameter: the column we want to compare with the price
    """
    plt.clf()
    plt.figure(figsize=(15, 10))
    sns.scatterplot(x=column,y='Price',data=df, hue='Property')
    plt.ticklabel_format(style='plain', axis='y')
    step = 500000
    max_price = df['Price'].max()  
    yticks = np.arange(0, max_price + step, step)
    plt.yticks(yticks, labels=[f"{int(y):,}" for y in yticks])
    plt.grid(alpha=0.5, linestyle='--')
    plt.savefig(f"./graphs/graph-{column}-price-property.png")
    plt.clf()
    sns.boxplot(data=df[column])
    plt.grid(alpha=0.5, linestyle='--')
    plt.savefig(f"./graphs/boxplot-{column}.png")

check_closer_corr('Garden surface')
check_closer_corr('Living area')
check_closer_corr('Surface of the plot')
check_closer_corr('Building condition')

# Creating boxplots to see outliers that may have to be removed
def check_boxplot(column: str):
    """
    Function that creates a boxplot based on one column and the price
    :Parameter: the column we want to compare with the price
    """
    plt.clf()
    plt.figure(figsize=(15, 15))
    sns.boxplot(x=column, y='Price', data=df)
    step = 500000
    max_price = df['Price'].max()  # Valeur maximum de Price
    yticks = np.arange(0, max_price + step, step)
    plt.yticks(yticks, labels=[f"{int(y):,}" for y in yticks]) 
    plt.xticks(rotation=90)    
    plt.savefig(f"./graphs/boxplot-{column}-price.png")

check_boxplot('Building condition')
check_boxplot('Property type')
check_boxplot('Property')

# Removing the outliers
def remove_outliers():
    """
    Functiun that removes the outliers we chose to be able to get the new correlation and info about the dataset
    """
    df.drop(df[df['Price'] > 2500000].index, inplace = True)
    df.drop(df[df['Property type'] == 'Other_Property'].index, inplace = True)
    df.drop(df[(df['Property type'] == 'Mixed_Use_Building') & (df['Living area'] > 1200)].index, inplace = True)
    df.drop(df[(df['Property'] == 'Apartment') & (df['Living area'] > 450)].index, inplace = True)
    df.drop(df[(df['Property'] == 'Apartment') & (df['Price'] > 1000000)].index, inplace = True)
    df.drop(df[(df['Property'] == 'House') & (df['Living area'] > 1200)].index, inplace = True)
    df.drop(df[(df['Property'] != 'House') & (df['Property'] != 'Apartment')].index, inplace = True)

remove_outliers()

# Creating the new correlation graphs to see the difference once the outliers have been removed
dataset_check_graphs_info("two")
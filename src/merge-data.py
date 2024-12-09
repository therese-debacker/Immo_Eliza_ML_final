import pandas as pd 
import numpy as np

# Import CSV files
df = pd.read_csv('./data/precleaned-dataset-immoweb.csv')
zip_code = pd.read_csv('./data/code-nis-zip-code.csv')
income_median = pd.read_csv('./data/median-income-2022.csv')
density_population = pd.read_csv('./data/density-population.csv')
income_mean = pd.read_csv('./data/mean-income-2022.csv')
surface_area = pd.read_csv('./data/surface-area-2024-district.csv',  header=None)
median_price = pd.read_csv('./data/sales-real-estates-belgium-district.csv')


# CLEANING NEW DATASETS
median_price = median_price.drop(median_price[(median_price['année'] != 2023)].index)
median_price.drop(columns= ['localité', 'année', 'période','prix premier quartile(€)-maison', 'prix troisième quartile(€)-maison','prix premier quartile(€)-appartement',
       'prix troisième quartile(€)-appartement', 'Unnamed: 12', 'Unnamed: 13',
       'Unnamed: 14', '0'], inplace=True)	
median_price = median_price.rename(columns={'nombre transactions - maison': 'nb_transactions_house','prix médian(€)-maison':'house-median-price',
                                            'nombre transactions-appartement':'nb_transactions_apartment','prix médian(€)-appartement':'apartment-median-price' })


median_price_refnis = median_price.groupby('refnis')['nb_transactions_house'].sum()
median_price['nb_transactions_house'] = median_price['refnis'].map(median_price_refnis)

median_price_refnis = median_price.groupby('refnis')['house-median-price'].mean()
median_price['house-median-price'] = median_price['refnis'].map(median_price_refnis)

median_price_refnis = median_price.groupby('refnis')['nb_transactions_apartment'].sum()
median_price['nb_transactions_apartment'] = median_price['refnis'].map(median_price_refnis)

median_price_refnis = median_price.groupby('refnis')['apartment-median-price'].mean()
median_price['apartment-median-price'] = median_price['refnis'].map(median_price_refnis)
median_price = median_price.drop_duplicates()


surface_area.columns = ['refnis','locality', 'rubrique', 'rubrique detail', 'number-parcels', 'surface-area-taxable', 'surface-area-exonerate', 
                        'surface-area-total', 'surface-area-promille']
surface_area.drop(columns= ['locality', 'surface-area-taxable', 'surface-area-exonerate', 'surface-area-promille'], inplace=True)	

surface_area_total = surface_area.drop(surface_area[(surface_area['rubrique'] != '6TOT')].index)
surface_area_built = surface_area.drop(surface_area[(surface_area['rubrique'] != '2TOT')].index)
surface_area_land = surface_area.drop(surface_area[(surface_area['rubrique'] != '1TOT')].index)

surface_area_total = surface_area_total.rename(columns={'number-parcels': 'number-parcels-total',})
surface_area_built = surface_area_built.rename(columns={'number-parcels': 'number-parcels-built','surface-area-total': 'surface-area-total-built'})
surface_area_land = surface_area_land.rename(columns={'number-parcels': 'number-parcels-land','surface-area-total': 'surface-area-total-land'})
surface_area_total = pd.merge(surface_area_total, surface_area_built, left_on='refnis', right_on='refnis', how='left')
surface_area_total = pd.merge(surface_area_total, surface_area_land, left_on='refnis', right_on='refnis', how='left')

surface_area_total.drop(columns= ['rubrique_x', 'rubrique detail_x', 'rubrique_y', 'rubrique detail_y', 'rubrique detail','rubrique'], inplace=True)	

income_median.drop(income_median[income_median['CD_YEAR'] != 2022].index, inplace = True)



# Merging CSVs
merged_df_income = pd.merge(zip_code, income_median, left_on='Refnis code', right_on='CD_MUNTY_REFNIS', how='left')
merged_df_population = pd.merge(merged_df_income, density_population, left_on='CD_MUNTY_REFNIS', right_on='code-ins', how='left')
merged_df_population.drop(columns=['men','women', 'code-ins'], inplace=True)
merged_df_avgincome = pd.merge(merged_df_population, income_mean, left_on='Nom commune', right_on='Nom', how='left')
merged_df_surface_area = pd.merge(merged_df_avgincome, surface_area_total, left_on='CD_DSTR_REFNIS', right_on='refnis', how='left')
merged_df_median_price = pd.merge(merged_df_surface_area, median_price, left_on='CD_DSTR_REFNIS', right_on='refnis', how='left')


# Removing unuseful columns
columns_to_drop = ['CD_RGN_REFNIS','TX_RGN_DESCR_NL','TX_RGN_DESCR_FR', 'TX_RGN_DESCR_EN', 'TX_RGN_DESCR_DE', 'CD_PROV_REFNIS', 'TX_PROV_DESCR_NL',
                            'TX_PROV_DESCR_FR', 'TX_PROV_DESCR_EN','TX_PROV_DESCR_DE','TX_DSTR_DESCR_NL','TX_DSTR_DESCR_FR','TX_DSTR_DESCR_EN','TX_DSTR_DESCR_DE',
                            'TX_MUNTY_DESCR_EN','TX_MUNTY_DESCR_DE','MS_Q1','MS_Q3','MS_NBR_ELIGIBLE','MS_NBR_NOT_ELIGIBLE','MS_PERC_NOT_ELIGIBLE','MS_PERC_IOE_HH',
                            'MS_INT_QUART_DIFF', 'Commune', 'CD_YEAR','TX_MUNTY_DESCR_NL','TX_MUNTY_DESCR_FR','Gemeentenaam', 'Nom commune','CD_MUNTY_REFNIS', 
                            'Refnis code', 'number-parcels-total', 'refnis_y','refnis_x']
merged_df_median_price.drop(columns=columns_to_drop, inplace=True)

merged_df_median_price = merged_df_median_price.rename(columns={'total': 'population','MS_MEDIAN': 'median-income','Nom': 'commune','Revenu': 'mean-income', 'MS_ADMIN_AROP': 'poverty-chance', 
                                    'CD_DSTR_REFNIS':'district'})


# Removing unnecessary elements
merged_df_median_price['poverty-chance'] = merged_df_median_price['poverty-chance'].str.replace(',', '.')
merged_df_median_price['population'] = merged_df_median_price['population'].str.replace(' ', '')

# Changing types
merged_df_median_price['poverty-chance'] = merged_df_median_price['poverty-chance'].astype(float)
merged_df_median_price['population'] = merged_df_median_price['population'].astype(float)
merged_df_median_price['district']=merged_df_median_price['district'].apply(str)

# Creating new columns
mean_income_district = merged_df_median_price.groupby('district')['mean-income'].mean()
merged_df_median_price['mean-income-district'] = merged_df_median_price['district'].map(mean_income_district)

median_income_district = merged_df_median_price.groupby('district')['median-income'].mean()
merged_df_median_price['median-income-district'] = merged_df_median_price['district'].map(median_income_district)

poverty_chance_district = merged_df_median_price.groupby('district')['poverty-chance'].mean()
merged_df_median_price['poverty-chance-district'] = merged_df_median_price['district'].map(poverty_chance_district)

population_district = merged_df_median_price.groupby('district')['population'].sum()
merged_df_median_price['population-district'] = merged_df_median_price['district'].map(population_district)

merged_df_median_price['population-per-surface-district'] = merged_df_median_price['population-district'] / merged_df_median_price['surface-area-total']

merged_df_median_price['number-transactions'] = merged_df_median_price['nb_transactions_house'] + merged_df_median_price['nb_transactions_apartment']


# Merging with the immoweb dataset
final_df = pd.merge(df, merged_df_median_price, left_on='Zip code', right_on='Postal code', how='left')
final_df.drop(columns=['Postal code'], inplace=True)

# Adding anew column
final_df['median-price'] = np.where(
    final_df['Property'] == 'House', 
    final_df['house-median-price'],       
    final_df['apartment-median-price']     
)


# Export CSV
final_df.to_csv('./data/merged-data.csv')


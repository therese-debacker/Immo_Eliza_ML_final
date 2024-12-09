import pandas as pd
from src.cleaning_datasets import CleaningDatasets
from src.cleaning_feature_engineering import DataCleaning
from src.linear_regression_model import LinearRegressionModel


def main():
    """
    Main script to clean, preprocess, and train a linear regression model
    for predicting real estate prices.
    """
    # Step 1: Import and clean datasets
    datapath = './data/'
    cleaner = CleaningDatasets(datapath)
    (
        df,
        zip_code,
        income_median,
        density_population,
        income_mean,
        surface_area,
        median_price,
    ) = cleaner.import_files()

    # Merge datasets
    merged_df_income = cleaner.merging_dataset(zip_code, income_median, 'Refnis code', 'CD_MUNTY_REFNIS')
    merged_df_population = cleaner.merging_dataset(merged_df_income, density_population, 'CD_MUNTY_REFNIS', 'code-ins')
    cleaner.drop_columns(merged_df_population, ['men', 'women', 'code-ins'])
    merged_df_avgincome = cleaner.merging_dataset(merged_df_population, income_mean, 'Nom commune', 'Nom')
    merged_df_surface_area = cleaner.merging_dataset(merged_df_avgincome, surface_area, 'CD_DSTR_REFNIS', 'refnis')
    merged_df_median_price = cleaner.merging_dataset(merged_df_surface_area, median_price, 'CD_DSTR_REFNIS', 'refnis')

    # Perform initial cleaning on the merged dataset
    columns_to_drop = [
        'CD_RGN_REFNIS', 'TX_RGN_DESCR_NL', 'TX_RGN_DESCR_FR', 'TX_RGN_DESCR_EN', 'TX_RGN_DESCR_DE',
        'CD_PROV_REFNIS', 'TX_PROV_DESCR_NL', 'TX_PROV_DESCR_FR', 'TX_PROV_DESCR_EN', 'TX_PROV_DESCR_DE',
        'TX_DSTR_DESCR_NL', 'TX_DSTR_DESCR_FR', 'TX_DSTR_DESCR_EN', 'TX_DSTR_DESCR_DE',
        'TX_MUNTY_DESCR_EN', 'TX_MUNTY_DESCR_DE', 'MS_Q1', 'MS_Q3', 'MS_NBR_ELIGIBLE',
        'MS_NBR_NOT_ELIGIBLE', 'MS_PERC_NOT_ELIGIBLE', 'MS_PERC_IOE_HH', 'MS_INT_QUART_DIFF',
        'Commune', 'CD_YEAR', 'TX_MUNTY_DESCR_NL', 'TX_MUNTY_DESCR_FR', 'Gemeentenaam',
        'Nom commune', 'CD_MUNTY_REFNIS', 'Refnis code'
    ]
    cleaner.drop_columns(merged_df_median_price, columns_to_drop)

    # Rename columns for consistency
    merged_df_median_price = cleaner.rename_columns(
        merged_df_median_price,
        {
            'total': 'population',
            'MS_MEDIAN': 'median-income',
            'Nom': 'commune',
            'Revenu': 'mean-income',
            'MS_ADMIN_AROP': 'poverty-chance',
            'CD_DSTR_REFNIS': 'district'
        }
    )

    # Final preprocessing: Replace elements and change types
    cleaner.replace_elements(merged_df_median_price, 'poverty-chance', ',', '.')
    cleaner.replace_elements(merged_df_median_price, 'population', ' ', '')
    cleaner.change_type(merged_df_median_price, 'poverty-chance', float)
    cleaner.change_type(merged_df_median_price, 'population', float)
    cleaner.change_type(merged_df_median_price, 'district', str)

    # Step 2: Clean and preprocess the main dataset
    final_df = cleaner.merging_dataset(df, merged_df_median_price, 'Zip code', 'Postal code')
    final_df.drop(columns=['Postal code'], inplace=True)
    final_df.to_csv(f'{datapath}merged-data.csv', index=False)

    # Apply feature engineering
    data_cleaner = DataCleaning(final_df)
    final_df = data_cleaner.remove_outliers()
    final_df = data_cleaner.remove_rows()
    final_df = data_cleaner.replace_navalues()
    final_df = data_cleaner.transforming_columns()
    columns_to_drop = [
        'Facades', 'Equipped kitchen', 'Furnished', 'Fireplace', 'Garden', 'Terrace',
        'Terrace surface', 'Region', 'Bedrooms', 'Locality', 'Garden surface',
        'poverty-chance', 'Property', 'surface-area-total', 'number-parcels-built'
    ]
    final_df = data_cleaner.removing_columns(columns_to_drop)
    categorical_columns = ['Property type', 'district']
    final_df = data_cleaner.transforming_categorical_values(categorical_columns)

    # Save the cleaned and preprocessed dataset
    final_df.to_csv(f'{datapath}dataset-preprocessed.csv', index=False)

    # Step 3: Train the linear regression model
    model_trainer = LinearRegressionModel(final_df)
    model_trainer.creation_linear_model()


if __name__ == "__main__":
    main()

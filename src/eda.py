import pandas as pd


def exploratory_data_analysis(df_final):

    df_final["Driver Top 3 Finish Percentage (This Year till last race)"] = df_final["Driver Top 3 Finish Percentage (This Year till last race)"].astype(float)
    df_final["Constructor Top 3 Finish Percentage (This Year till last race)"] = df_final["Constructor Top 3 Finish Percentage (This Year till last race)"].astype(float)
    df_final["Driver Average Position (This Year till last race)"] = df_final["Driver Average Position (This Year till last race)"].astype(float)
    df_final["Constructor Average Position (This Year till last race)"] = df_final["Constructor Average Position (This Year till last race)"].astype(float)

    # Use describe to select Data analytics and specific rows
    description = df_final.describe()
    selected_description = description.loc[['count', 'mean', 'std', 'min', 'max']]

    # Get the Correlations
    correlations = df_final.corr()['Top 3 Finish'].sort_values(ascending=False)

    #Converting categorical data into machine learning format 
    df_final_encoded = pd.get_dummies(df_final, columns=['circuitId', 'driverId', 'constructorId'])

    # Creating a list of Columns excluding the one to move
    cols = [col for col in df_final_encoded.columns if col != 'Top 3 Finish']

    # Append the column to the end of the DataFrame
    df_final_encoded = df_final_encoded[cols + ['Top 3 Finish']]

    return df_final_encoded
import pandas as pd

def load_data(folder_path):
    constructors_df = pd.read_csv(folder_path + '/constructors.csv')
    drivers_df = pd.read_csv(folder_path + '/drivers.csv')
    races_df = pd.read_csv(folder_path + '/races.csv')
    results_df = pd.read_csv(folder_path + '/results.csv')
    
    return constructors_df, drivers_df, races_df, results_df


def preprocess_data(folder_path):
    pd.set_option('display.max_columns', None)
    constructors_df, drivers_df, races_df, results_df = load_data(folder_path)
    
    # Data merging, new columns creation
    race_df = races_df[["raceId", "year", "round", "circuitId"]].copy()
    race_df = race_df.sort_values(by=['year', 'round'])
    race_df = race_df[race_df["year"] >= 1982]
    res_df = results_df[['raceId', 'driverId', 'constructorId', 'grid', 'positionOrder']].copy()


    df = pd.merge(race_df, res_df, on='raceId')

    
    return df, drivers_df, constructors_df


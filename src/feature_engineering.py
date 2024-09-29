import pandas as pd

def calculate_driver_features(df):
    df['Top 3 Finish'] = df['positionOrder'].le(3).astype(int)
    # We do this to calculate the drivers yearly stats
    driver_yearly_stats = df.groupby(['year', 'driverId']).agg(
        Total_Races=('raceId', 'nunique'),
        Top_3_Finishes=('Top 3 Finish', 'sum')
    ).reset_index()
    # This calculates  percentage of top 3 finished for each driver in each year
    driver_yearly_stats['Driver Top 3 Finish Percentage (This Year)'] = (driver_yearly_stats['Top_3_Finishes'] / driver_yearly_stats['Total_Races']) * 100
    driver_last_year_stats = driver_yearly_stats.copy()
    driver_last_year_stats['year'] += 1
    driver_last_year_stats = driver_last_year_stats.rename(columns={'Driver Top 3 Finish Percentage (This Year)': 'Driver Top 3 Finish Percentage (Last Year)'})
    
    df = pd.merge(df, driver_last_year_stats[['year', 'driverId', 'Driver Top 3 Finish Percentage (Last Year)']], on=['year', 'driverId'], how='left')

    #Calculates the Mean of top 3 finished percentages for 2 drivers in each constructor last year
    constructor_last_year_stats = df.groupby(['year', 'constructorId', 'round']).agg(
        Sum_Top_3_Finishes_Last_Year=('Driver Top 3 Finish Percentage (Last Year)', 'sum')
    ).reset_index()

    # This calculates the perentage of top 3 finished for each constructor last year
    constructor_last_year_stats['Constructor Top 3 Finish Percentage (Last Year)'] = constructor_last_year_stats["Sum_Top_3_Finishes_Last_Year"]/2
    df = pd.merge(df, constructor_last_year_stats[['year', 'constructorId', 'round', 'Constructor Top 3 Finish Percentage (Last Year)']], on=['year', 'constructorId', 'round'], how='left')

    # A function that calculates the top 3 finish percentage before the current round of drivers
    def calculate_driver_top_3_percentage_before_round(row, df):
        # This filters the races for the same driver in the same year but earlier rounds
        previous_races = df[(df['year'] == row['year']) & (df['driverId'] == row['driverId']) & (df['round'] < row['round'])]
        if len(previous_races) == 0:
            return pd.NA

        total_races = previous_races['raceId'].nunique()
        top_3_finishes = previous_races['Top 3 Finish'].sum()

        # Calculates the percentage
        return (top_3_finishes / total_races) * 100 if total_races > 0 else pd.NA

    # We Then Apply the function to each row in the DataFrame
    df['Driver Top 3 Finish Percentage (This Year till last race)'] = df.apply(lambda row: calculate_driver_top_3_percentage_before_round(row, df), axis=1)

    #The same but for 2 drivers in each constructor this year
    constructor_this_year_stats = df.groupby(['year', 'constructorId', 'round']).agg(
        Sum_Top_3_Finishes_This_Year=('Driver Top 3 Finish Percentage (This Year till last race)', 'sum')
    ).reset_index()

    #Same but for each constructor not driver
    constructor_this_year_stats['Constructor Top 3 Finish Percentage (This Year till last race)'] = constructor_this_year_stats["Sum_Top_3_Finishes_This_Year"]/2

    df = pd.merge(df, constructor_this_year_stats[['year', 'constructorId', 'round', 'Constructor Top 3 Finish Percentage (This Year till last race)']], on=['year', 'constructorId', 'round'], how='left')

    #Calculates total num of races and top 3 finished for each driver each year
    driver_yearly_stats = df.groupby(['year', 'driverId']).agg(
        Total_Races=('raceId', 'nunique'),
        Avg_position=('positionOrder', 'mean')
    ).reset_index()

    # We then calculate percentage of top 3 finished for a driver in a year
    driver_yearly_stats['Driver Avg position (This Year)'] = driver_yearly_stats['Avg_position']

    # Shift percentages for driver from last year to next year 
    driver_last_year_stats = driver_yearly_stats.copy()
    driver_last_year_stats['year'] += 1
    driver_last_year_stats = driver_last_year_stats.rename(columns={'Driver Avg position (This Year)': 'Driver Avg position (Last Year)'})

    df = pd.merge(df, driver_last_year_stats[['year', 'driverId', 'Driver Avg position (Last Year)']], on=['year', 'driverId'], how='left')


    # Calculates the mean of top 3 finished percentages for 2 drivers in each constructor last year
    constructor_last_year_stats = df.groupby(['year', 'constructorId', 'round']).agg(
        sum_position_last_year=('Driver Avg position (Last Year)', 'sum')
    ).reset_index()

    # Calc the percentage of top 3 finished for each constructor last year
    constructor_last_year_stats['Constructor Avg position (Last Year)'] = constructor_last_year_stats["sum_position_last_year"]/2

    df = pd.merge(df, constructor_last_year_stats[['year', 'constructorId', 'round', 'Constructor Avg position (Last Year)']], on=['year', 'constructorId', 'round'], how='left')

    # This function calculates the drivers average position before the round
    def calculate_driver_avg_position_before_round(row, df):
        # Races in teh same year and driver but earlier rounds again
        previous_races = df[(df['year'] == row['year']) & (df['driverId'] == row['driverId']) & (df['round'] < row['round'])]
        if len(previous_races) == 0:
            return pd.NA
        # Calculate Total Races and E of positions
        total_races = previous_races['raceId'].nunique()
        positionSum = previous_races['positionOrder'].sum()

        # Average Position, Simple Mean Calc
        return (positionSum / total_races) if total_races > 0 else pd.NA

    # Countinue by applying the funtion to each row
    df['Driver Average Position (This Year till last race)'] = df.apply(lambda row: calculate_driver_avg_position_before_round(row, df), axis=1)

    #Calculates mean of top 3 finished percentages for 2 drivers in each contructor as done before
    constructor_this_year_stats = df.groupby(['year', 'constructorId', 'round']).agg(
        sum_Position_Constructor = ('Driver Average Position (This Year till last race)', 'sum')
    ).reset_index()

    #Percentage of top 3 finished for each constructor this year
    constructor_this_year_stats['Constructor Average Position (This Year till last race)'] = constructor_this_year_stats["sum_Position_Constructor"]/2

    df = pd.merge(df, constructor_this_year_stats[['year', 'constructorId', 'round', 'Constructor Average Position (This Year till last race)']], on=['year', 'constructorId', 'round'], how='left')

    return df

def feature_selection(df):
    # Drop information related to the result of the race to prevent leakage
    df_final = df.drop(labels=["raceId"], axis=1)

    # Count rows where 'year' is not 1982 so we can drop NaN values
    initial_count = len(df_final[df_final['year'] != 1982])

    # Drop Nan rows
    df_final = df_final.dropna()

    # Count rows Which were not dropped
    final_count = len(df_final[df_final['year'] != 1982])

    # Calculate the number of rows dropped
    rows_dropped = initial_count - final_count

    df_final_keepPositionOrder = df_final.copy()
    df_final = df_final.drop(["positionOrder"], axis = 1)

    return df_final

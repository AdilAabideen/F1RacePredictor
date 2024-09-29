import pandas as pd
import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from src.data_preprocessing import load_data

pd.set_option('display.max_columns', None)

folder_path = 'data'
constructors_df, drivers_df, races_df, results_df = load_data(folder_path)

# Extract only relevant information about the races
race_df = races_df[["raceId", "year", "round", "circuitId"]].copy()

# Sort the race_df by 'year' and 'round'
race_df = race_df.sort_values(by=['year', 'round'])

# Filter out races before 1982
race_df = race_df[race_df["year"] >= 1982]


# Extract relevant columns from results_df
res_df = results_df[['raceId', 'driverId', 'constructorId', 'grid', 'positionOrder']].copy()

# Check for duplicate rows in race_df
duplicates = race_df.duplicated()
num_duplicates = duplicates.sum()

df = pd.merge(race_df, res_df, on='raceId')

df['Top 3 Finish'] = df['positionOrder'].le(3).astype(int)

# Calculate driver yearly stats (Total Races and Top 3 Finishes)
driver_yearly_stats = df.groupby(['year', 'driverId']).agg(
    Total_Races=('raceId', 'nunique'),
    Top_3_Finishes=('Top 3 Finish', 'sum')
).reset_index()


# Calculating the percentage of top 3 finishes for each driver in each year
driver_yearly_stats['Driver Top 3 Finish Percentage (This Year)'] = (driver_yearly_stats['Top_3_Finishes'] / driver_yearly_stats['Total_Races']) * 100

# Shifting the driver percentages to the next year for last year's data
driver_last_year_stats = driver_yearly_stats.copy()
driver_last_year_stats['year'] += 1
driver_last_year_stats = driver_last_year_stats.rename(columns={'Driver Top 3 Finish Percentage (This Year)': 'Driver Top 3 Finish Percentage (Last Year)'})

df = pd.merge(df, driver_last_year_stats[['year', 'driverId', 'Driver Top 3 Finish Percentage (Last Year)']], on=['year', 'driverId'], how='left')

# Checking the merged data
constructor_last_year_stats = df.groupby(['year', 'constructorId', 'round']).agg(
    Sum_Top_3_Finishes_Last_Year=('Driver Top 3 Finish Percentage (Last Year)', 'sum')
).reset_index()

# print("Constructor annual stats")
# print(constructor_last_year_stats)

# Calculating the percentage of top 3 finishes for each constructor last year
constructor_last_year_stats['Constructor Top 3 Finish Percentage (Last Year)'] = constructor_last_year_stats["Sum_Top_3_Finishes_Last_Year"]/2

df = pd.merge(df, constructor_last_year_stats[['year', 'constructorId', 'round', 'Constructor Top 3 Finish Percentage (Last Year)']], on=['year', 'constructorId', 'round'], how='left')

# Checking the merged data
# print("New dataframe")
# print(df[df["year"]>=1983])


# Function to calculate the percentage of Top 3 finishes before the current race
def calculate_driver_top_3_percentage_before_round(row, df):
    # Filter for races in the same year, for the same driver, but in earlier rounds
    previous_races = df[(df['year'] == row['year']) & 
                        (df['driverId'] == row['driverId']) & 
                        (df['round'] < row['round'])]
    if len(previous_races) == 0:
        return pd.NA

    total_races = previous_races['raceId'].nunique()
    top_3_finishes = previous_races['Top 3 Finish'].sum()

    # Calculate the percentage
    return (top_3_finishes / total_races) * 100 if total_races > 0 else pd.NA

# Apply the function to each row in the DataFrame
df['Driver Top 3 Finish Percentage (This Year till last race)'] = df.apply(lambda row: calculate_driver_top_3_percentage_before_round(row, df), axis=1)

# Optionally, print the new dataframe or its subset to check
# print(df[['driverId', 'year', 'round', 'Driver Top 3 Finish Percentage (This Year till last race)']].head())
# Calculating mean of top 3 finishes percentages for the two drivers in each constructor this year
constructor_this_year_stats = df.groupby(['year', 'constructorId', 'round']).agg(
    Sum_Top_3_Finishes_This_Year=('Driver Top 3 Finish Percentage (This Year till last race)', 'sum')
).reset_index()

# print("Constructor annual stats")
# print(constructor_this_year_stats)

# Calculating the percentage of top 3 finishes for each constructor this year
constructor_this_year_stats['Constructor Top 3 Finish Percentage (This Year till last race)'] = constructor_this_year_stats["Sum_Top_3_Finishes_This_Year"]/2

df = pd.merge(df, constructor_this_year_stats[['year', 'constructorId', 'round', 'Constructor Top 3 Finish Percentage (This Year till last race)']], on=['year', 'constructorId', 'round'], how='left')

# Checking the merged data
# print("New dataframe")
# print(df[df["year"]>=1983])


# Calculating the total number of races and top 3 finishes for each driver in each year
driver_yearly_stats = df.groupby(['year', 'driverId']).agg(
    Total_Races=('raceId', 'nunique'),
    Avg_position=('positionOrder', 'mean')
).reset_index()

# print("Driver annual stats")
# print(driver_yearly_stats)

# Calculating the percentage of top 3 finishes for each driver in each year
driver_yearly_stats['Driver Avg position (This Year)'] = driver_yearly_stats['Avg_position']

# Shifting the driver percentages to the next year for last year's data
driver_last_year_stats = driver_yearly_stats.copy()
driver_last_year_stats['year'] += 1
driver_last_year_stats = driver_last_year_stats.rename(columns={'Driver Avg position (This Year)': 'Driver Avg position (Last Year)'})

df = pd.merge(df, driver_last_year_stats[['year', 'driverId', 'Driver Avg position (Last Year)']], on=['year', 'driverId'], how='left')

# Checking the merged data
# print("New dataframe")
# print(df[df["year"]>=1983])

constructor_last_year_stats = df.groupby(['year', 'constructorId', 'round']).agg(
    sum_position_last_year=('Driver Avg position (Last Year)', 'sum')
).reset_index()

# print("Constructor annual stats")
# print(constructor_last_year_stats)

# Calculating the percentage of top 3 finishes for each constructor last year
constructor_last_year_stats['Constructor Avg position (Last Year)'] = constructor_last_year_stats["sum_position_last_year"]/2

df = pd.merge(df, constructor_last_year_stats[['year', 'constructorId', 'round', 'Constructor Avg position (Last Year)']], on=['year', 'constructorId', 'round'], how='left')

# Checking the merged data
# print("New dataframe")
# print(df[df["year"]>=1983])

def calculate_driver_avg_position_before_round(row, df):
    # Filter for races in the same year, for the same driver, but in earlier rounds
    previous_races = df[(df['year'] == row['year']) & (df['driverId'] == row['driverId']) & (df['round'] < row['round'])]
    if len(previous_races) == 0:
      return pd.NA
    # Calculate the total races and sum of positions
    total_races = previous_races['raceId'].nunique()
    positionSum = previous_races['positionOrder'].sum()

    # Calculate average position
    return (positionSum / total_races) if total_races > 0 else pd.NA

# Apply the function to each row in the DataFrame
df['Driver Average Position (This Year till last race)'] = df.apply(lambda row: calculate_driver_avg_position_before_round(row, df), axis=1)

constructor_this_year_stats = df.groupby(['year', 'constructorId', 'round']).agg(
    sum_Position_Constructor = ('Driver Average Position (This Year till last race)', 'sum')
).reset_index()

# print("Constructor annual stats")
# print(constructor_this_year_stats)

# Calculating the percentage of top 3 finishes for each constructor this year
constructor_this_year_stats['Constructor Average Position (This Year till last race)'] = constructor_this_year_stats["sum_Position_Constructor"]/2

df = pd.merge(df, constructor_this_year_stats[['year', 'constructorId', 'round', 'Constructor Average Position (This Year till last race)']], on=['year', 'constructorId', 'round'], how='left')

# Checking the merged data
# print("New dataframe")
# print(df[df["year"]>=1983])


# print(df[(df["year"] == 2023)& (df["round"] > 3) ].head(30))
nan_counts = df.isna().sum()
# print(nan_counts)


df_final = df.drop(labels=["raceId"], axis=1)
# print("Number of rows in total:", df_final.shape[0])

# Count rows where 'year' is not 1982 before dropping NaN values
initial_count = len(df_final[df_final['year'] != 1982])

# Drop rows with NaN values
df_final = df_final.dropna()

# Count rows where 'year' is not 1982 after dropping NaN values
final_count = len(df_final[df_final['year'] != 1982])

# Calculate the number of rows dropped
rows_dropped = initial_count - final_count

# print("Number of rows dropped where year is not 1982:", rows_dropped)
df_final_keepPositionOrder = df_final.copy()
df_final = df_final.drop(["positionOrder"], axis = 1)
# print(df_final)

df_final["Driver Top 3 Finish Percentage (This Year till last race)"] = df_final["Driver Top 3 Finish Percentage (This Year till last race)"].astype(float)
df_final["Constructor Top 3 Finish Percentage (This Year till last race)"] = df_final["Constructor Top 3 Finish Percentage (This Year till last race)"].astype(float)
df_final["Driver Average Position (This Year till last race)"] = df_final["Driver Average Position (This Year till last race)"].astype(float)
df_final["Constructor Average Position (This Year till last race)"] = df_final["Constructor Average Position (This Year till last race)"].astype(float)

# Using describe() and selecting specific rows
description = df_final.describe()
selected_description = description.loc[['count', 'mean', 'std', 'min', 'max']]

# print(selected_description)

# heatmap
# plt.figure(figsize=(10,7))
# sns.heatmap(df_final.corr(), annot=True, mask = False, annot_kws={"size": 7})
# plt.show()

correlations = df_final.corr()['Top 3 Finish'].sort_values(ascending=False)

# Display
# print(correlations)

df_final_encoded = pd.get_dummies(df_final, columns=['circuitId', 'driverId', 'constructorId'])

# Create a list of columns excluding the one to move
cols = [col for col in df_final_encoded.columns if col != 'Top 3 Finish']

# Append the column to the end of the DataFrame
df_final_encoded = df_final_encoded[cols + ['Top 3 Finish']]

print(df_final_encoded)
print(df_final_encoded.shape)
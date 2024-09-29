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
from src.data_preprocessing import preprocess_data
from src.feature_engineering import calculate_driver_features
from src.feature_engineering import feature_selection
from src.eda import exploratory_data_analysis
from src.model_training import train_model
from src.model_training import model_interpretation
from src.model_training import model_interpretation_importance
from src.model_training import model_accuracy
from src.model_training import predict

pd.set_option('display.max_columns', None)

folder_path = 'data'

df, drivers_df, constructors_df = preprocess_data(folder_path)
df = calculate_driver_features(df)
df_final = feature_selection(df)
df_final_encoded = exploratory_data_analysis(df_final)
# train_model(df_final_encoded) 

# model_interpretation(df_final_encoded)
# model_interpretation_importance(df_final_encoded)

# model_accuracy(df_final_encoded)

round = int(input("Please enter the round "))


predict(round, df_final_encoded, df_final, drivers_df, constructors_df)









# Formula 1 Prediction System

This project aims to predict the likelihood of Formula 1 drivers and constructors finishing in the top 3 in a given race round, based on historical data. The system leverages machine learning techniques and incorporates extensive data preprocessing, feature engineering, exploratory data analysis (EDA), and model training using various classification algorithms.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Training](#model-training)
- [Model Interpretation](#model-interpretation)
- [Prediction](#prediction)
- [Future Improvements](#future-improvements)

## Project Structure

```bash
├── data
│   ├── constructors.csv
│   ├── drivers.csv
│   ├── races.csv
│   └── results.csv
├── src
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── eda.py
│   ├── model_training.py
│   └── init.py
├── main.py
└── README.md
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/f1-prediction-system.git
   cd f1-prediction-system
   ```
2.	Install the required dependencies:
```bash
pip install -r requirements.txt
```
3.	Ensure you have your data files in the data folder:
- constructors.csv
- drivers.csv
- races.csv
- results.csv

## Usage

1. To run the model, execute main.py:
   ```bash
   python main.py
   ```
2.	Input the race round when prompted, and the model will predict the top 3 finish likelihood for each driver.

## Data Preprocessing

In the preprocessing stage, data from multiple sources (constructors.csv, drivers.csv, races.csv, and results.csv) is merged to create a comprehensive dataset. We also filter out data before 1982 for consistency.

	•	Merging race and result data.
	•	Sorting races by year and round.
	•	Removing unnecessary columns.

Refer to src/data_preprocessing.py for details.

## Feature Engineering

Several new features are calculated, focusing on driver and constructor performance, such as:

	•	Driver top 3 finish percentage.
	•	Constructor top 3 finish percentage.
	•	Driver and constructor average positions.

These features are calculated for both the current year (up to the last race) and the previous year.

Refer to src/feature_engineering.py for the implementation.

## Exploratory Data Analysis

In the EDA step, we convert categorical data (e.g., driverId, constructorId) into machine-learning-compatible format and calculate correlations between various features and the target variable (Top 3 Finish).

Refer to src/eda.py for implementation details.

## Model Training

The project supports training multiple machine learning models:

	•	Logistic Regression
	•	K-Nearest Neighbors (KNN)
	•	Support Vector Classifier (SVC)
	•	Random Forest
	•	Decision Tree
	•	Naive Bayes

For each model, hyperparameters are tuned, and the model is evaluated using AUC-ROC scores. The best model is saved for future predictions.

To train the models, run train_model(df_final_encoded) in src/model_training.py.

## Model Interpretation

Model interpretation includes visualizing decision trees and identifying feature importance for the Random Forest model.

To interpret the model, run:
```bash
model_interpretation(df_final_encoded)
model_interpretation_importance(df_final_encoded)
```

Refer to src/model_training.py for more details.

## Prediction

The prediction system uses the trained Random Forest classifier to predict whether a driver will finish in the top 3 for a specified round of the current year.

To make predictions, run predict(round, df_final_encoded, df_final, drivers_df, constructors_df) in main.py.

## Future Improvements

	•	Enhance feature engineering with more advanced historical performance metrics.
	•	Experiment with other models like XGBoost.
	•	Deploy the model as a web app with a user-friendly interface.

For more details, refer to the source code in the src/ directory

```bash
Feel free to copy and paste this `README.md` into your project folder! Let me know if you need further adjustments.
```

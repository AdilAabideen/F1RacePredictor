import pandas as pd
import matplotlib
import numpy as np
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, ConfusionMatrixDisplay  # Import f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from joblib import load
from sklearn.metrics import precision_recall_curve, f1_score


def train_model(df_final_encoded):
    
    # Drop uneeded Data
    df_1 = df_final_encoded.drop(["grid", "Driver Top 3 Finish Percentage (This Year till last race)", "Constructor Top 3 Finish Percentage (This Year till last race)", "Driver Average Position (This Year till last race)", "Constructor Average Position (This Year till last race)", "Driver Average Position (This Year till last race)", "Constructor Average Position (This Year till last race)"], axis = 1)
    # Add more data by dropping less
    df_2 = df_final_encoded.drop(["grid", "Driver Average Position (This Year till last race)", "Constructor Average Position (This Year till last race)", "Driver Average Position (This Year till last race)", "Constructor Average Position (This Year till last race)"], axis = 1)

    # Finally we only drop Grid 
    df_3 = df_final_encoded.drop(["grid"], axis = 1)

    # We then dont drop any data
    df_with_qualifying = df_final_encoded

    # Since we are trying to find top 3 there is a mismatch between what we want and dont want , 17 v 3 so we use AUC_ROC score instead of accuracy
    train_df = df_with_qualifying[(df_with_qualifying["year"] >= 1983) & (df_with_qualifying["year"] <= 2008)]
    val_df = df_with_qualifying[(df_with_qualifying["year"] >= 2009) & (df_with_qualifying["year"] <= 2016)]
    test_df = df_with_qualifying[(df_with_qualifying["year"] >= 2017) & (df_with_qualifying["year"] <= 2023)]

    X_train = train_df[train_df.columns.tolist()[:-1]].values
    y_train = train_df['Top 3 Finish'].values
    X_val = val_df[train_df.columns.tolist()[:-1]].values
    y_val = val_df['Top 3 Finish'].values
    X_test = test_df[train_df.columns.tolist()[:-1]].values
    y_test = test_df['Top 3 Finish'].values

    # This includes Percetnage of Finishing in top 3 for this year up until last race for drivers and for constructors
    # And includes Average finishing performance of drivers and constructors last year, average finishing positions of drivers and constructors this year up till last race


    #////////

    # We use a Dictionary to store the best model and test accuracy for each algorithm
    model_accuracy_info = {}

    # We tehn fefine the hyperprameter grid for each model
    param_grid = {
        'LogisticRegression': {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'random_state': [42]},
        'KNeighborsClassifier': {'n_neighbors': [3, 5, 7, 10, 12, 13, 15, 20]},
        'SVC': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear'], 'gamma': ['scale', 'auto'], 'random_state': [42], 'probability': [True]},
        'RandomForestClassifier': {'n_estimators': [10, 50, 100], 'max_depth': [None, 10, 20, 30], 'random_state': [42]},
        'DecisionTreeClassifier': {'max_depth': [None, 5, 10, 20], 'random_state': [42]},
        'GaussianNB': {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}
    }

    # Initialize all the models
    models = {
        'LogisticRegression': LogisticRegression(),
        'KNeighborsClassifier': KNeighborsClassifier(),
        'SVC': SVC(),
        'RandomForestClassifier': RandomForestClassifier(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'GaussianNB': GaussianNB(),
    }

    # Ignore all convergence and any future warnings
    from warnings import simplefilter
    from sklearn.exceptions import ConvergenceWarning
    simplefilter("ignore", category=ConvergenceWarning)
    simplefilter("ignore", category=FutureWarning)

    # This function manually tunes the parameters
    def tune_hyperparameters(model, params, X_train, y_train, X_val, y_val):
        best_model = None
        best_params = {}
        best_auc = 0  # Use AUC-ROC instead of F1 score
        for param in ParameterGrid(params):
            model.set_params(**param)
            model.fit(X_train, y_train)
            probabilities = model.predict_proba(X_val)
            auc = roc_auc_score(y_val, probabilities[:, 1])  # Calculate AUC-ROC
            if auc > best_auc:
                best_auc = auc
                best_model = model
                best_params = param
        return best_model, best_params

    # tuning and evaluation of perform hyperprameter 
    for name, model in models.items():
        print(f"Model: {name}")
        if name in param_grid:
            # Tune hyperparameters
            best_model, best_params = tune_hyperparameters(model, param_grid[name], X_train, y_train, X_val, y_val)
            print(f"Best parameters for {name}: {best_params}")
            model = best_model

        # Predict and evaluate on test data using AUC-ROC again
        pred_test = model.predict_proba(X_test)
        auc_test = roc_auc_score(y_test, pred_test[:, 1])  # Calculate AUC-ROC
        accuracy_test = accuracy_score(y_test, pred_test[:, 1] >= 0.5)  # Calculate accuracy

        print(f"Test AUC-ROC for {name}: {auc_test:.4f}\n")
        model_filename = f"{name}_model_V4.joblib"
        joblib.dump(model, model_filename)
        model_accuracy_info[name] = {
            'model': model_filename,
            'auc_roc': auc_test,  # Store AUC-ROC
            'accuracy': accuracy_test
        }

        # Calculate ROC curve and AUC for each model 
        fpr, tpr, thresholds = roc_curve(y_test, pred_test[:, 1])
        roc_auc = auc(fpr, tpr)

        # Plot the ROC curve
        plt.plot(fpr, tpr, label=f'{name} (area = {roc_auc:.2f})')

        # Print model and F1 score info
    for model_name, info in model_accuracy_info.items():
        print(f"Model: {model_name}, File: {info['model']}, Test AUC-ROC: {info['auc_roc']:.4f}, Test Accuracy: {info['accuracy']:.4f}")

    plt.plot([0, 1], [0, 1], 'k--')  # Add a diagonal dashed line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.show()

def model_interpretation(df_final_encoded) :

    df_with_qualifying = df_final_encoded

    # Since we are trying to find top 3 there is a mismatch between what we want and dont want , 17 v 3 so we use AUC_ROC score instead of accuracy
    train_df = df_with_qualifying[(df_with_qualifying["year"] >= 1983) & (df_with_qualifying["year"] <= 2008)]
    
    #We define the maximum deoth to display
    max_depth = 2
    dt_model = joblib.load("DecisionTreeClassifier_model_V4.joblib")

    # Set the size of the plot we want
    plt.figure(figsize=(70, 20))

    # Plot the tree with maximum depth defined above
    plot_tree(dt_model,
            max_depth=max_depth,
            filled=True,
            rounded=True,
            class_names=['Class 0', 'Class 1'],
            feature_names=train_df.columns,
            fontsize = 30)

    # Show the plot to us
    plt.show()

    dt_feature_importances = dt_model.feature_importances_

    # Creating a DataFrame from the feature names and their importances
    dt_importances_df = pd.DataFrame({
        'Feature': train_df.columns[:-1],  
        'Importance': dt_feature_importances
    })

    # Sorting those dataframes by importance
    dt_importances_df.sort_values(by='Importance', ascending=False, inplace=True)

    print(dt_importances_df.head(10))

def model_interpretation_importance(df_final_encoded):

    df_with_qualifying = df_final_encoded

    model = load("RandomForestClassifier_model_V4.joblib")
    importances = model.feature_importances_

    # Create a DataFrame for visualization
    importances_df = pd.DataFrame({'Feature': df_with_qualifying.columns[:-1], 'Importance': importances})

    # Sort the DataFrame
    importances_df.sort_values(by='Importance', ascending=False, inplace=True)

    # Display the feature importances
    print(importances_df.head(10))

    print(model.get_params(deep = True))


def model_accuracy(df_final_encoded):
    df_with_qualifying = df_final_encoded
    train_df = df_with_qualifying[(df_with_qualifying["year"] >= 1983) & (df_with_qualifying["year"] <= 2008)]
    test_df = df_with_qualifying[(df_with_qualifying["year"] >= 2017) & (df_with_qualifying["year"] <= 2023)]
    X_test = test_df[train_df.columns.tolist()[:-1]].values
    y_test = test_df['Top 3 Finish'].values
    model = load("RandomForestClassifier_model_V4.joblib")

    # Predict probabilities on new data
    data_prob = model.predict_proba(X_test)[:, 1]

    # Compute Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, data_prob)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)

    # Find the optimal threshold
    optimal_idx = np.nanargmax(f1_scores)  # Using nanargmax to ignore NaN values
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal Threshold for F1 Score: {optimal_threshold:.3f}")

    # Apply the new threshold to make class predictions
    pred_test = (data_prob >= optimal_threshold).astype(int)

    # Display the Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, pred_test, display_labels=['Class 0', 'Class 1'])
    plt.title("Confusion Matrix for Random Forest Classifier on Test Data")
    plt.show()

    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test, pred_test)
    print(f"Accuracy of the model: {accuracy:.3f}")

    # Calculate and print the F1 score
    f1 = f1_score(y_test, pred_test)
    print(f"F1 Score of the model: {f1:.3f}")

    # Create a dictionary to hold the model and the threshold
    model_data = {
        "model": model,
        "threshold": optimal_threshold
    }

    # Save the dictionary
    model_filename = "RandomForestClassifier_model_V4_with_threshold.joblib"
    joblib.dump(model_data, model_filename)

def predict(round, df_final_encoded, df_final, drivers_df, constructors_df):
    df_with_qualifying = df_final_encoded
    round = round
    # Load the model and threshold
    model_data = load("RandomForestClassifier_model_V4_with_threshold.joblib")
    model = model_data["model"]
    optimal_threshold = model_data["threshold"]

    # Filter the DataFrame for the year 2023 and round 7 (Spanish Grand Prix)
    df_2023 = df_with_qualifying[(df_with_qualifying["year"] == 2023) & (df_with_qualifying["round"] == round)]

    # Prepare the feature matrix for prediction
    X_2023 = df_2023[df_2023.columns.tolist()[:-1]].values

    # Compute probabilities using the model
    probabilities = model.predict_proba(X_2023)

    # Apply the threshold to make class predictions
    pred_test = (probabilities[:, 1] >= optimal_threshold).astype(int)  # Assuming the second column represents the probability of top 3 finish

    # Selecting relevant columns and filtering for year 2023 and round 7
    df_predict2023 = df_final[["year", "round", "driverId", "constructorId", "grid", "Top 3 Finish"]]
    df_predict2023 = df_predict2023[(df_predict2023["year"] == 2023) & (df_predict2023["round"] == round)]



    # Create a dictionary to map 'driverId' to 'surname'
    driver_name_dict = pd.Series(drivers_df.surname.values, index=drivers_df.driverId).to_dict()

    # Replace 'driverId' with corresponding 'surname' in df_predict2023
    df_predict2023['driverId'] = df_predict2023['driverId'].map(driver_name_dict)

    # Append the predictions to the DataFrame
    df_predict2023['Top_3_Finish_Prediction'] = pred_test


    # Create a dictionary to map 'constructorId' to 'name'
    constructor_name_dict = pd.Series(constructors_df.name.values, index=constructors_df.constructorId).to_dict()

    # Replace 'constructorId' with corresponding 'name' in df_predict2023
    df_predict2023['constructorId'] = df_predict2023['constructorId'].map(constructor_name_dict)

    # Append the predictions to the DataFrame
    df_predict2023['Top_3_Finish_Prediction'] = pred_test


    # Print the DataFrame with predictions
    print(df_predict2023)
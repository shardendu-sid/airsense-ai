from models import model
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV, learning_curve, KFold
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import joblib

current_directory = os.path.dirname(os.path.abspath(__file__))


# load_processed_data_model = os.environ.get('load_processed_data')

temp_model_save_path = "/Users/shardendujha/thesis-project-final/src/temp_random_forest_model.pkl"
humid_model_save_path = "/Users/shardendujha/thesis-project-final/src/humid_random_forest_model.pkl"
co2_model_save_path = "/Users/shardendujha/thesis-project-final/src/co2_random_forest_model.pkl"
voc_model_save_path = "/Users/shardendujha/thesis-project-final/src/voc_random_forest_model.pkl"
pm25_model_save_path = "/Users/shardendujha/thesis-project-final/src/pm25_random_forest_model.pkl"

trained_models = {}

def temp_pred_model():
    # Load the collected data into a DataFrame
    row_data = os.environ.get("CSV_FILE")
    awair_csv_data = pd.read_csv(row_data, low_memory=False).tail(288)
    selected_required_columns = (
        awair_csv_data[["temp", "humid", "co2", "voc", "pm25"]]
        .dropna()
        .drop_duplicates()
    )

    # Split the data into input features (X) and target variable (y)
    X = selected_required_columns.drop("temp", axis=1)
    y = selected_required_columns["temp"]

    # Time-based Train-Test Split (No Random Shuffle)
    train_size = int(0.8 * len(X))  # 80% training, 20% future test
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # # Split the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42
    # )

    # Train Random Forest Regressor
    model_temp = RandomForestRegressor(
        n_estimators=500,           # Increased number of trees
        max_depth=15,               # Control tree depth to avoid overfitting
        max_features="log2",        # Use sqrt for better feature selection
        min_samples_split=5,        # Make splits a bit more general
        min_samples_leaf=2,         # Keep at least 2 samples in leaves
        random_state=42
    )

    # # Fit the model on the training data
    model_temp.fit(X_train, y_train)

    # Make predictions using the trained model on the test set.
    y_train_pred = model_temp.predict(X_train)
    y_test_pred = model_temp.predict(X_test)

    # Calculate evaluation metrics (RMSE, MAE, R2 score)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))  # RMSE
    train_mae = mean_absolute_error(y_train, y_train_pred)  # MAE
    train_r2 = r2_score(y_train, y_train_pred)  # R2 score

    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))  # RMSE
    test_mae = mean_absolute_error(y_test, y_test_pred)  # MAE
    test_r2 = r2_score(y_test, y_test_pred)  # R2 score

    # Round the metrics to 2 decimal places
    train_rmse_rounded = round(train_rmse, 2)
    train_mae_rounded = round(train_mae, 2)
    train_r2_rounded = round(train_r2, 2)

    test_rmse_rounded = round(test_rmse, 2)
    test_mae_rounded = round(test_mae, 2)
    test_r2_rounded = round(test_r2, 2)

    trained_models.update(
        {"temp_train_rmse": train_rmse_rounded,
         "temp_test_rmse": test_rmse_rounded,
        "temp_train_mae":train_mae_rounded,
        "temp_test_mae": test_mae_rounded,
        "temp_train_r2": train_r2_rounded,
        "temp_test_r2": test_r2_rounded}
    )

    print("==== Temperature Model Performance ====")
    print(f"TRAIN RMSE: {train_rmse_rounded}")
    print(f"TEST RMSE: {test_rmse_rounded}")

    print(f"TRAIN MAE: {train_mae_rounded}")
    print(f"TEST MAE: {test_mae_rounded}")

    print(f"TRAIN R2 Score: {train_r2_rounded}")
    print(f"TEST R2 Score: {test_r2_rounded}")

    joblib.dump(model_temp, temp_model_save_path)
    print("===============================")
    print("Temperature Model trained and saved successfully!")
    print("Next training scheduled in 24 hours.")

    # === Visualization (Actual vs. Predicted) ===
    plt.figure(figsize=(12, 6))

    # Train set plot
    plt.subplot(1, 2, 1)
    plt.plot(y_train.values[:100], label="Actual", color="blue")
    plt.plot(y_train_pred[:100], label="Predicted", color="red", linestyle="dashed")
    plt.title("Train Set: Actual vs. Predicted")
    plt.xlabel("Sample Index")
    plt.ylabel("Temperature")
    plt.legend()

    # Test set plot
    plt.subplot(1, 2, 2)
    plt.plot(y_test.values[:100], label="Actual", color="blue")
    plt.plot(y_test_pred[:100], label="Predicted", color="red", linestyle="dashed")
    plt.title("Test Set: Actual vs. Predicted")
    plt.xlabel("Sample Index")
    plt.ylabel("Temperature")
    plt.legend()

    plt.tight_layout()
    plt.savefig('/Users/shardendujha/thesis-project-final/notebook/temp_model_performance.png')  # Save the plot as an image
    print("Plot saved as /home/pi/temp_model_performance.png")

    
    return (
        model_temp,
        train_rmse_rounded,
        test_rmse_rounded,
        train_mae_rounded,
        test_mae_rounded,
        train_r2_rounded,
        test_r2_rounded,
    )


def humid_pred_model():

    row_data = os.environ.get("CSV_FILE")
    awair_csv_data = pd.read_csv(row_data, low_memory=False).tail(288)
    selected_required_columns = (
        awair_csv_data[["temp", "humid", "co2", "voc", "pm25"]]
        .dropna()
        .drop_duplicates()
    )
    
    
    # We are predicting 'humid' using features ['temp', 'co2', 'pm25']
    X = selected_required_columns[["temp", "co2", "voc", "pm25"]]
    y = selected_required_columns["humid"]

    # Time-based Train-Test Split (No Random Shuffle)
    train_size = int(0.8 * len(X))  # 80% training, 20% future test
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42
    # )

    model_humid = RandomForestRegressor(
        n_estimators=200,       # More trees for stability
        max_depth=5,           # Allow deeper trees for better fit
        max_features=None,    # Try log2 or None for better splits
        min_samples_split=25,    # Allow earlier splits
        min_samples_leaf=2,     # Allow smaller leaf nodes
        random_state=42         # Seed for reproducibility
    )

    model_humid.fit(X_train, y_train)

    # Make predictions using the trained model on the test set.
    y_train_pred = model_humid.predict(X_train)
    y_test_pred = model_humid.predict(X_test)
    # print(f"Y_pred : {y_pred}")

    # Calculate evaluation metrics (RMSE, MAE, R2 score)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))  # RMSE
    train_mae = mean_absolute_error(y_train, y_train_pred)  # MAE
    train_r2 = r2_score(y_train, y_train_pred)  # R2 score

    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))  # RMSE
    test_mae = mean_absolute_error(y_test, y_test_pred)  # MAE
    test_r2 = r2_score(y_test, y_test_pred)  # R2 score

    # Round the metrics to 2 decimal places
    train_rmse_rounded = round(train_rmse, 2)
    train_mae_rounded = round(train_mae, 2)
    train_r2_rounded = round(train_r2, 2)

    test_rmse_rounded = round(test_rmse, 2)
    test_mae_rounded = round(test_mae, 2)
    test_r2_rounded = round(test_r2, 2)

    trained_models.update(
        {"humid_train_rmse": train_rmse_rounded,
         "humid_test_rmse": test_rmse_rounded,
        "humid_train_mae": train_mae_rounded,
        "humid_test_mae": test_mae_rounded,
        "humid_train_r2": train_r2_rounded,
        "humid_test_r2": test_r2_rounded}
    )

    print("==== Humidity Model Performance ====")
    print(f"TRAIN RMSE: {train_rmse_rounded}")
    print(f"TEST RMSE: {test_rmse_rounded}")

    print(f"TRAIN MAE: {train_mae_rounded}")
    print(f"TEST MAE: {test_mae_rounded}")

    print(f"TRAIN R2 Score: {train_r2_rounded}")
    print(f"TEST R2 Score: {test_r2_rounded}")

    joblib.dump(model_humid, humid_model_save_path)
    print("===============================")
    print("Humidity Model trained and saved successfully!")
    print("Next training scheduled in 24 hours.")

    # === Visualization (Actual vs. Predicted) ===
    plt.figure(figsize=(12, 6))

    # Train set plot
    plt.subplot(1, 2, 1)
    plt.plot(y_train.values[:100], label="Actual", color="blue")
    plt.plot(y_train_pred[:100], label="Predicted", color="red", linestyle="dashed")
    plt.title("Train Set: Actual vs. Predicted")
    plt.xlabel("Sample Index")
    plt.ylabel("Humidity")
    plt.legend()

    # Test set plot
    plt.subplot(1, 2, 2)
    plt.plot(y_test.values[:100], label="Actual", color="blue")
    plt.plot(y_test_pred[:100], label="Predicted", color="red", linestyle="dashed")
    plt.title("Test Set: Actual vs. Predicted")
    plt.xlabel("Sample Index")
    plt.ylabel("Humidity")
    plt.legend()

    plt.tight_layout()
    plt.savefig('/Users/shardendujha/thesis-project-final/notebook/humid_model_performance.png')  # Save the plot as an image
    print("Plot saved as /Users/shardendujha/thesis-project-final/notebook/humid_model_performance.png")

    return (
        train_rmse_rounded,
        test_rmse_rounded,
        train_mae_rounded,
        test_mae_rounded,
        train_r2_rounded,
        test_r2_rounded,
    )


def co2_pred_model():

    # Load the collected data into a DataFrame
    row_data = os.environ.get("CSV_FILE")
    awair_csv_data = pd.read_csv(row_data, low_memory=False).tail(288)
    selected_required_columns = (
        awair_csv_data[["temp", "humid", "co2", "voc", "pm25"]]
        .dropna()
        .drop_duplicates()
    )
    # Split the data into input features (X) and target variable (y)

    # Split the data into input features (X) and target variable (y)
    X = selected_required_columns[["temp", "humid", "voc", "pm25"]]
    y = selected_required_columns["co2"]

    # Time-based Train-Test Split (No Random Shuffle)
    train_size = int(0.8 * len(X))  # 80% training, 20% future test
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # # Split the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42
    # )
    # Initialize and train the RandomForest model
    model_co2 = RandomForestRegressor(
        n_estimators=200,           # Increased number of trees
        max_depth=5,               # Control tree depth to avoid overfitting
        max_features=None,        # Use sqrt for better feature selection
        min_samples_split=15,        # Make splits a bit more general
        min_samples_leaf=2,         # Keep at least 2 samples in leaves
        random_state=42
    )

    # # Fit the model on the training data
    model_co2.fit(X_train, y_train)

    
    # Make predictions using the trained model on the test set.
    y_train_pred = model_co2.predict(X_train)
    y_test_pred = model_co2.predict(X_test)


    # Calculate evaluation metrics (RMSE, MAE, R2 score)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))  # RMSE
    train_mae = mean_absolute_error(y_train, y_train_pred)  # MAE
    train_r2 = r2_score(y_train, y_train_pred)  # R2 score

    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))  # RMSE
    test_mae = mean_absolute_error(y_test, y_test_pred)  # MAE
    test_r2 = r2_score(y_test, y_test_pred)  # R2 score

    # Round the metrics to 2 decimal places
    train_rmse_rounded = round(train_rmse, 2)
    train_mae_rounded = round(train_mae, 2)
    train_r2_rounded = round(train_r2, 2)

    test_rmse_rounded = round(test_rmse, 2)
    test_mae_rounded = round(test_mae, 2)
    test_r2_rounded = round(test_r2, 2)

    trained_models.update(
        {"co2_train_rmse": train_rmse_rounded,
         "co2_test_rmse": test_rmse_rounded,
        "co2_train_mae": train_mae_rounded,
        "co2_test_mae": test_mae_rounded,
        "co2_train_r2": train_r2_rounded,
        "co2_test_r2": test_r2_rounded}
    )

    print("==== Co2 Model Performance ====")
    print(f"TRAIN RMSE: {train_rmse_rounded}")
    print(f"TEST RMSE: {test_rmse_rounded}")

    print(f"TRAIN MAE: {train_mae_rounded}")
    print(f"TEST MAE: {test_mae_rounded}")

    print(f"TRAIN R2 Score: {train_r2_rounded}")
    print(f"TEST R2 Score: {test_r2_rounded}")

    joblib.dump(model_co2, co2_model_save_path)
    print("===============================")
    print("Co2 Model trained and saved successfully!")
    print("Next training scheduled in 24 hours.")

    # === Visualization (Actual vs. Predicted) ===
    plt.figure(figsize=(12, 6))

    # Train set plot
    plt.subplot(1, 2, 1)
    plt.plot(y_train.values[:100], label="Actual", color="blue")
    plt.plot(y_train_pred[:100], label="Predicted", color="red", linestyle="dashed")
    plt.title("Train Set: Actual vs. Predicted")
    plt.xlabel("Sample Index")
    plt.ylabel("Co2")
    plt.legend()

    # Test set plot
    plt.subplot(1, 2, 2)
    plt.plot(y_test.values[:100], label="Actual", color="blue")
    plt.plot(y_test_pred[:100], label="Predicted", color="red", linestyle="dashed")
    plt.title("Test Set: Actual vs. Predicted")
    plt.xlabel("Sample Index")
    plt.ylabel("Co2")
    plt.legend()

    plt.tight_layout()
    plt.savefig('/Users/shardendujha/thesis-project-final/notebook/co2_model_performance.png')  # Save the plot as an image
    print("Plot saved as /Users/shardendujha/thesis-project-final/notebook/co2_model_performance.png")

    return (
        train_rmse_rounded,
        test_rmse_rounded,
        train_mae_rounded,
        test_mae_rounded,
        train_r2_rounded,
        test_r2_rounded

    )


def voc_pred_model():
    # Load the collected data into a DataFrame
    row_data = os.environ.get("CSV_FILE")
    awair_csv_data = pd.read_csv(row_data, low_memory=False).tail(288)
    selected_required_columns = (
        awair_csv_data[["temp", "humid", "co2", "voc", "pm25"]]
        .dropna()
        .drop_duplicates()
    )
    # Split the data into input features (X) and target variable (y)

    # Split the data into input features (X) and target variable (y)
    X = selected_required_columns[["temp", "humid", "co2", "pm25"]]
    y = selected_required_columns["voc"]

    # Time-based Train-Test Split (No Random Shuffle)
    train_size = int(0.8 * len(X))  # 80% training, 20% future test
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Initialize and train the RandomForest model
    model_voc = RandomForestRegressor(
        n_estimators=300,       # More trees for stability
        max_depth=12,           # Allow deeper trees for better fit
        max_features='log2',    # Try log2 or None for better splits
        min_samples_split=8,    # Allow earlier splits
        min_samples_leaf=2,     # Allow smaller leaf nodes
        random_state=42         # Seed for reproducibility
    )

    # # Fit the model on the training data
    model_voc.fit(X_train, y_train)

    # Make predictions using the trained model on the test set.
    y_train_pred = model_voc.predict(X_train)
    y_test_pred = model_voc.predict(X_test)

    # Calculate evaluation metrics (RMSE, MAE, R2 score)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))  # RMSE
    train_mae = mean_absolute_error(y_train, y_train_pred)  # MAE
    train_r2 = r2_score(y_train, y_train_pred)  # R2 score

    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))  # RMSE
    test_mae = mean_absolute_error(y_test, y_test_pred)  # MAE
    test_r2 = r2_score(y_test, y_test_pred)  # R2 score

    # Round the metrics to 2 decimal places
    train_rmse_rounded = round(train_rmse, 2)
    train_mae_rounded = round(train_mae, 2)
    train_r2_rounded = round(train_r2, 2)

    test_rmse_rounded = round(test_rmse, 2)
    test_mae_rounded = round(test_mae, 2)
    test_r2_rounded = round(test_r2, 2)

    trained_models.update(
        {"voc_train_rmse": train_rmse_rounded,
         "voc_test_rmse": test_rmse_rounded,
        "voc_train_mae": train_mae_rounded,
        "voc_test_mae": test_mae_rounded,
        "voc_train_r2": train_r2_rounded,
        "voc_test_r2": test_r2_rounded}
    )

    print("==== VOC Model Performance ====")
    print(f"TRAIN RMSE: {train_rmse_rounded}")
    print(f"TEST RMSE: {test_rmse_rounded}")

    print(f"TRAIN MAE: {train_mae_rounded}")
    print(f"TEST MAE: {test_mae_rounded}")

    print(f"TRAIN R2 Score: {train_r2_rounded}")
    print(f"TEST R2 Score: {test_r2_rounded}")

    joblib.dump(model_voc, voc_model_save_path)
    print("===============================")
    print("VOC Model trained and saved successfully!")
    print("Next training scheduled in 24 hours.")

    # === Visualization (Actual vs. Predicted) ===
    plt.figure(figsize=(12, 6))

    # Train set plot
    plt.subplot(1, 2, 1)
    plt.plot(y_train.values[:100], label="Actual", color="blue")
    plt.plot(y_train_pred[:100], label="Predicted", color="red", linestyle="dashed")
    plt.title("Train Set: Actual vs. Predicted")
    plt.xlabel("Sample Index")
    plt.ylabel("VOC")
    plt.legend()

    # Test set plot
    plt.subplot(1, 2, 2)
    plt.plot(y_test.values[:100], label="Actual", color="blue")
    plt.plot(y_test_pred[:100], label="Predicted", color="red", linestyle="dashed")
    plt.title("Test Set: Actual vs. Predicted")
    plt.xlabel("Sample Index")
    plt.ylabel("VOC")
    plt.legend()

    plt.tight_layout()
    plt.savefig('/Users/shardendujha/thesis-project-final/notebook/voc_model_performance.png')  # Save the plot as an image
    print("Plot saved as /Users/shardendujha/thesis-project-final/notebook/voc_model_performance.png")


    return (
        train_rmse_rounded,
        test_rmse_rounded,
        train_mae_rounded,
        test_mae_rounded,
        train_r2_rounded,
        test_r2_rounded,
    )

def pm25_pred_model():
    # Load and preprocess the data
    row_data = os.environ.get("CSV_FILE")
    awair_csv_data = pd.read_csv(row_data, low_memory=False).tail(588)  # Use recent 288 rows
    selected_required_columns = awair_csv_data[["temp", "humid", "co2", "voc", "pm25"]].dropna().drop_duplicates()

    # #Feature engineering
    # # Add lag features (PM2.5 from the past 1 and 2 hours)
    # selected_required_columns['pm25_lag_1'] = selected_required_columns['pm25'].shift(1)
    # selected_required_columns['pm25_lag_2'] = selected_required_columns['pm25'].shift(2)
    
    # # Add interaction features
    # selected_required_columns['temp_humid'] = selected_required_columns['temp'] * selected_required_columns['humid']
    # selected_required_columns['voc_co2'] = selected_required_columns['voc'] * selected_required_columns['co2']
    
    # Drop any rows with NaN values created by the lag features
    selected_required_columns = selected_required_columns.dropna()

    # Split data into input features (X) and target variable (y)
    X = selected_required_columns.drop("pm25", axis=1)  # Drop pm25 as it's the target
    y = selected_required_columns["pm25"]

    # Time-based Train-Test Split (No Random Shuffle)
    train_size = int(0.8 * len(X))  # 80% training, 20% future test
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # # Split the data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the RandomForest model
    model_pm25 = RandomForestRegressor(
        n_estimators=200,       # Reduce trees for stability and training speed
        max_depth=40,           # Limit depth to avoid overfitting
        max_features=None,    # More features considered at each split
        min_samples_split=2,    # Allow earlier splits
        min_samples_leaf=2,     # Leaf nodes must have at least 5 samples
        random_state=42         # Seed for reproducibility
    )

    model_pm25.fit(X_train, y_train)

    # Make predictions using the trained model on train and test sets
    y_train_pred = model_pm25.predict(X_train)
    y_test_pred = model_pm25.predict(X_test)

    # Calculate evaluation metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Round the metrics to 2 decimal places for better readability
    train_rmse_rounded = round(train_rmse, 2)
    train_mae_rounded = round(train_mae, 2)
    train_r2_rounded = round(train_r2, 2)
    test_rmse_rounded = round(test_rmse, 2)
    test_mae_rounded = round(test_mae, 2)
    test_r2_rounded = round(test_r2, 2)

    trained_models.update(
        {"pm25_train_rmse": train_rmse_rounded,
         "pm25_test_rmse": test_rmse_rounded,
        "pm25_train_mae": train_mae_rounded,
        "pm25_test_mae": test_mae_rounded,
        "pm25_train_r2": train_r2_rounded,
        "pm25_test_r2": test_r2_rounded}
    )

    # Print model performance
    print("==== Pm2.5 Model Performance ====")
    print(f"TRAIN RMSE: {train_rmse_rounded}")
    print(f"TEST RMSE: {test_rmse_rounded}")
    print(f"TRAIN MAE: {train_mae_rounded}")
    print(f"TEST MAE: {test_mae_rounded}")
    print(f"TRAIN R2 Score: {train_r2_rounded}")
    print(f"TEST R2 Score: {test_r2_rounded}")
    print("===============================")
    
    # Save the trained model for later use
    # pm25_model_save_path = "pm25_model.pkl"
    joblib.dump(model_pm25, pm25_model_save_path)
    print("Pm2.5 Model trained and saved successfully!")
    print("Next training scheduled in 24 hours.")
    
    # === Visualization (Actual vs. Predicted) ===
    plt.figure(figsize=(12, 6))

    # Train set plot
    plt.subplot(1, 2, 1)
    plt.plot(y_train.values[:100], label="Actual", color="blue")
    plt.plot(y_train_pred[:100], label="Predicted", color="red", linestyle="dashed")
    plt.title("Train Set: Actual vs. Predicted")
    plt.xlabel("Sample Index")
    plt.ylabel("Pm2.5")
    plt.legend()

    # Test set plot
    plt.subplot(1, 2, 2)
    plt.plot(y_test.values[:100], label="Actual", color="blue")
    plt.plot(y_test_pred[:100], label="Predicted", color="red", linestyle="dashed")
    plt.title("Test Set: Actual vs. Predicted")
    plt.xlabel("Sample Index")
    plt.ylabel("Pm2.5")
    plt.legend()

    plt.tight_layout()
    plt.savefig('/Users/shardendujha/thesis-project-final/notebook/pm25_model_performance.png')  # Save the plot as an image
    print("Plot saved as /Users/shardendujha/thesis-project-final/notebook/pm25_model_performance.png")


    return (
        train_rmse_rounded,
        test_rmse_rounded,
        train_mae_rounded,
        test_mae_rounded,
        train_r2_rounded,
        test_r2_rounded,
    )


# temp_pred_model()
# humid_pred_model()
# co2_pred_model()
# voc_pred_model()
# pm25_pred_model()
# print(trained_models)

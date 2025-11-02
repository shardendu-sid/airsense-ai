import sys
sys.path.insert(0, 'src/models')
from model import temp_pred_model,humid_pred_model,co2_pred_model,voc_pred_model,pm25_pred_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import os
load_processed_data_model = os.environ.get("load_processed_data")

def plot_temp_prediction():
     # Load the collected data into a DataFrame
    awair_csv_data = pd.read_csv(load_processed_data_model)
    selected_required_columns = awair_csv_data
    # Split the data into input features (X) and target variable (y)
    X = selected_required_columns.drop('temp', axis=1)
    y = selected_required_columns ['temp']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Initialize and train the linear regression model
    model_temp = temp_pred_model()
    # # Fit the model on the training data
    model_temp.fit(X_train, y_train)

    #Make predictions using the trained model on the test set.

    y_pred = model_temp.predict(X_test)

    # Plotting the actual values
    plt.plot(y_test, label='Actual')
    # Plotting the predicted values
    plt.plot(y_pred, label='Predicted')
    # Adding labels and title to the plot
    plt.xlabel('Data Point')
    plt.ylabel('Temperature Value')
    plt.title('Actual vs Predicted temperature Values')
    # Adding legend
    plt.legend()
    # Display the plot
    plt.show()

def plot_humid_prediction():
    
    # Load the collected data into a DataFrame
    awair_csv_data = pd.read_csv(load_processed_data_model)
    selected_required_columns = awair_csv_data

    # Split the data into input features (X) and target variable (y)
    X = selected_required_columns.drop('humid', axis=1)
    y = selected_required_columns ['humid']


    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Initialize and train the linear regression model
    model_humid = humid_pred_model()
    # # Fit the model on the training data
    model_humid.fit(X_train, y_train)

    #Make predictions using the trained model on the test set.

    y_pred = model_humid.predict(X_test)
    # Plotting the actual values
    plt.plot(y_test, label='Actual')
    # Plotting the predicted values
    plt.plot(y_pred, label='Predicted')
    # Adding labels and title to the plot
    plt.xlabel('Data Point')
    plt.ylabel('Humidity Value')
    plt.title('Actual vs Predicted humid Values')
    # Adding legend
    plt.legend()
    # Display the plot
    plt.show()


def plot_co2_prediction():

    # Load the collected data into a DataFrame
    awair_csv_data = pd.read_csv(load_processed_data_model)
    selected_required_columns = awair_csv_data
    
    # Split the data into input features (X) and target variable (y)
    X = selected_required_columns.drop('co2', axis=1)
    y = selected_required_columns ['co2']


    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Initialize and train the RandomForest model
    model_co2 = co2_pred_model()

    # # Fit the model on the training data
    model_co2.fit(X_train, y_train)

    #Make predictions using the trained model on the test set.

    y_pred = model_co2.predict(X_test)
    # Plotting the actual values
    plt.plot(y_test, label='Actual')
    # Plotting the predicted values
    plt.plot(y_pred, label='Predicted')
    # Adding labels and title to the plot
    plt.xlabel('Data Point')
    plt.ylabel('CO2 Value')
    plt.title('Actual vs Predicted CO2 Values')
    # Adding legend
    plt.legend()

    # Display the plot
    plt.show()

def plot_voc_prediction():
    # Load the collected data into a DataFrame
    awair_csv_data = pd.read_csv(load_processed_data_model)
    selected_required_columns = awair_csv_data

    # Split the data into input features (X) and target variable (y)
    X = selected_required_columns.drop('voc', axis=1)
    y = selected_required_columns ['voc']


    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Initialize and train the RandomForest model
    model_voc = voc_pred_model()
    # # Fit the model on the training data
    model_voc.fit(X_train, y_train)

    #Make predictions using the trained model on the test set.

    y_pred = model_voc.predict(X_test)
    # Plotting the actual values
    plt.plot(y_test, label='Actual')
    # Plotting the predicted values
    plt.plot(y_pred, label='Predicted')
    # Adding labels and title to the plot
    plt.xlabel('Data Point')
    plt.ylabel('VOC Value')
    plt.title('Actual vs Predicted voc Values')
    # Adding legend
    plt.legend()
    # Display the plot
    plt.show()

def plot_pm25_prediction():

    # Load the collected data into a DataFrame
    awair_csv_data = pd.read_csv(load_processed_data_model)
    selected_required_columns = awair_csv_data
    
    # Split the data into input features (X) and target variable (y)
    X = selected_required_columns.drop('pm25', axis=1)
    y = selected_required_columns ['pm25']


    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Initialize and train the RandomForest model
    model_pm25 = pm25_pred_model()
    # # Fit the model on the training data
    model_pm25.fit(X_train, y_train)

    #Make predictions using the trained model on the test set.

    y_pred = model_pm25.predict(X_test)
    # Plotting the actual values
    plt.plot(y_test, label='Actual')
    # Plotting the predicted values
    plt.plot(y_pred, label='Predicted')
    # Adding labels and title to the plot
    plt.xlabel('Data Point')
    plt.ylabel('PM2.5 Value')
    plt.title('Actual vs Predicted Pm2.5 Values')
    # Adding legend
    plt.legend()

    # Display the plot
    plt.show()


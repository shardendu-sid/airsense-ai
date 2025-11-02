import sys
import os
import pandas as pd
import requests
import pytz
from datetime import datetime as dt
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import joblib

# Get the directory of the current file
current_dir = os.path.dirname(os.path.realpath(__file__))

# Get the parent directory (backend/src)
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to sys.path
sys.path.append(parent_dir)

# from train import (
#     temp_pred_model,
#     humid_pred_model,
#     co2_pred_model,
#     voc_pred_model,
#     pm25_pred_model,
# )


url = os.environ.get("API_URL")

predicted_data = []


def live_sensor_data():

    sensor_data = []
    Url = url  # api url path
    request = requests.request("GET", Url)
    data = request.json()
    add_new_col = {"location": "Janonhanta1,Vantaa,Finland"}

    add_bew_col_serial = {}
    data.update(add_bew_col_serial)
    data.update(add_new_col)
    sensor_data.append(data)

    return sensor_data


def temp_test_prediction():

    df = pd.DataFrame(live_sensor_data())
    selected_real_time_data = df[["humid", "co2", "voc", "pm25"]].astype(float)

    selected_real_time_data = selected_real_time_data.dropna()
    selected_real_time_data = selected_real_time_data.drop_duplicates()


    # Convert current time to Helsinki timezone
    helsinki_timezone = pytz.timezone("Europe/Helsinki")
    current_time_local = dt.now(helsinki_timezone).strftime("%Y-%m-%d %H:%M:%S")

    # current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    predicted_data.append({"DateTime": current_time_local})
   

    # Load the trained model 
    model_temp = joblib.load("/Users/shardendujha/thesis-project-final/src/temp_random_forest_model.pkl")  # Load your pre-trained model
    
    # Create input example (with realistic values or zeros)
    input_example = pd.DataFrame(
        {"humid": [48.78], "co2": [599.0], "voc": [154.0], "pm25": [1.0]}
    )

    with mlflow.start_run():
        signature = mlflow.models.infer_signature(
            selected_real_time_data, model_temp.predict(selected_real_time_data)
        )

        mlflow.sklearn.log_model(
            model_temp,
            "temperature-prediction-model",
            input_example=input_example,
            signature=signature,
        )
        
        # Make predictions using the model
        prediction_temperature = model_temp.predict(selected_real_time_data)
        extracted_value = round(prediction_temperature[0], 2)

        # Log the prediction (optional)
        mlflow.log_metric("Predicted_Temperature", extracted_value)


    predicted_data.append({"Temp_Pred": extracted_value})
    print(f"predicted_temperature: {extracted_value}")

    return extracted_value


def humid_test_prediction():
    # Predict humidity for new data
    df = pd.DataFrame(live_sensor_data())
    selected_real_time_data = df[["temp", "co2", "voc" ,"pm25"]].astype(float)

    selected_real_time_data = selected_real_time_data.dropna()
    selected_real_time_data = selected_real_time_data.drop_duplicates()

    # Convert current time to Helsinki timezone
    helsinki_timezone = pytz.timezone("Europe/Helsinki")
    current_time_local = dt.now(helsinki_timezone).strftime("%Y-%m-%d %H:%M:%S")

    predicted_data.append({"DateTime": current_time_local})
    
    # Load the trained model 
    model_humid = joblib.load("/Users/shardendujha/thesis-project-final/src/humid_random_forest_model.pkl")  # Load your pre-trained model
    input_example = pd.DataFrame(
            {"temp": [48.78], "co2": [599.0],"voc": [154.0], "pm25": [1.0]}
        )
    
    with mlflow.start_run():
        signature = mlflow.models.infer_signature(
            selected_real_time_data, model_humid.predict(selected_real_time_data)
        )

        mlflow.sklearn.log_model(
            model_humid,
            "temperature-prediction-model",
            input_example=input_example,
            signature=signature,
        )

        # Make predictions using the model
        prediction_humidity = model_humid.predict(selected_real_time_data)
        extracted_value = round(prediction_humidity[0], 2)

        # Log the prediction (optional)
        mlflow.log_metric("Predicted_Humidity", extracted_value)

    predicted_data.append({"Humid_Pred": extracted_value})
    print(f"predicted_humidity: {extracted_value}")
    
    return extracted_value


def co2_test_prediction():
    # Predict CO2 for new data

    df2 = pd.DataFrame(live_sensor_data())
    selected_real_time_data_co2 = df2[["temp", "humid", "voc", "pm25"]].astype(float)

    selected_real_time_data_co2 = selected_real_time_data_co2.dropna()
    selected_real_time_data_co2 = selected_real_time_data_co2.drop_duplicates()
    # print("===========================================================")

    # Convert current time to Helsinki timezone
    helsinki_timezone = pytz.timezone("Europe/Helsinki")
    current_time_local = dt.now(helsinki_timezone).strftime("%Y-%m-%d %H:%M:%S")

    # current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    predicted_data.append({"DateTime": current_time_local})
    # Load the trained model 
    model_co2 = joblib.load("/Users/shardendujha/thesis-project-final/src/co2_random_forest_model.pkl")  # Load your pre-trained model
    input_example = pd.DataFrame(
            {"temp": [48.78], "humid": [599.0], "voc": [154.0],"pm25": [1.0]}
        )
    
    with mlflow.start_run():
        signature = mlflow.models.infer_signature(
            selected_real_time_data_co2, model_co2.predict(selected_real_time_data_co2)
        )

        mlflow.sklearn.log_model(
            model_co2,
            "temperature-prediction-model",
            input_example=input_example,
            signature=signature,
        )

        # Make predictions using the model
        prediction_co2 = model_co2.predict(selected_real_time_data_co2)
        extracted_value = round(prediction_co2[0], 2)

        # Log the prediction (optional)
        mlflow.log_metric("Predicted_Co2", extracted_value)

    predicted_data.append({"Co2_Pred": extracted_value})
    print(f"predicted_co2: {extracted_value}")
    
    return extracted_value


def voc_test_prediction():
    # Predict VOC for new data

    df3 = pd.DataFrame(live_sensor_data())
    selected_real_time_data_voc = df3[["temp", "humid", "co2", "pm25"]].astype(float)

    selected_real_time_data_voc = selected_real_time_data_voc.dropna()
    selected_real_time_data_voc = selected_real_time_data_voc.drop_duplicates()
    # print("===========================================================")

    # Convert current time to Helsinki timezone
    helsinki_timezone = pytz.timezone("Europe/Helsinki")
    current_time_local = dt.now(helsinki_timezone).strftime("%Y-%m-%d %H:%M:%S")

    # current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    predicted_data.append({"DateTime": current_time_local})
    # Load the trained model 
    model_voc = joblib.load("/Users/shardendujha/thesis-project-final/src/voc_random_forest_model.pkl")  # Load your pre-trained model
    input_example = pd.DataFrame(
            {"temp": [48.78], "humid": [599.0], "co2": [154.0], "pm25": [1.0]}
        )
    
    with mlflow.start_run():
        signature = mlflow.models.infer_signature(
            selected_real_time_data_voc, model_voc.predict(selected_real_time_data_voc)
        )

        mlflow.sklearn.log_model(
            model_voc,
            "temperature-prediction-model",
            input_example=input_example,
            signature=signature,
        )

        # Make predictions using the model
        prediction_voc = model_voc.predict(selected_real_time_data_voc)
        extracted_value = round(prediction_voc[0], 2)

         # Log the prediction (optional)
        mlflow.log_metric("Predicted_Voc", extracted_value)

    predicted_data.append({"Voc_Pred": extracted_value})
    print(f"predicted_Voc: {extracted_value}")
    return extracted_value


def pm25_test_prediction():
    # Predict PM2.5 for new data

    df4 = pd.DataFrame(live_sensor_data())
    selected_real_time_data_pm25 = df4[["temp", "humid", "co2", "voc"]].astype(float)

    selected_real_time_data_pm25 = selected_real_time_data_pm25.dropna()
    selected_real_time_data_pm25 = selected_real_time_data_pm25.drop_duplicates()

    # Convert current time to Helsinki timezone
    helsinki_timezone = pytz.timezone("Europe/Helsinki")
    current_time_local = dt.now(helsinki_timezone).strftime("%Y-%m-%d %H:%M:%S")

    # current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    predicted_data.append({"DateTime": current_time_local})
    # Load the trained model 
    model_pm25 = joblib.load("/Users/shardendujha/thesis-project-final/src/pm25_random_forest_model.pkl")  # Load your pre-trained model
    input_example = pd.DataFrame(
            {"temp": [48.78], "humid": [599.0], "co2": [154.0], "voc": [1.0]}
        )
    
    with mlflow.start_run():
        signature = mlflow.models.infer_signature(
            selected_real_time_data_pm25, model_pm25.predict(selected_real_time_data_pm25)
        )

        mlflow.sklearn.log_model(
            model_pm25,
            "temperature-prediction-model",
            input_example=input_example,
            signature=signature,
        )

        # Make predictions using the model
        prediction_pm25 = model_pm25.predict(selected_real_time_data_pm25)
        extracted_value = round(prediction_pm25[0], 2)

         # Log the prediction (optional)
        mlflow.log_metric("Predicted_Pm25", extracted_value)

    predicted_data.append({"Pm25_Pred": extracted_value})
    print(f"predicted_Pm25: {extracted_value}")
    

    return extracted_value


# def compare_live_predicted():
#     # Example predicted and live data (use your actual data)
#     predicted_data = {
#         'Temp_Pred': [21.81],
#         'Humid_Pred': [29.03],
#         'Co2_Pred': [616.96],
#         'Voc_Pred': [134.71],
#         'Pm25_Pred': [1.96]
#     }

#     live_data = {
#         'temp': [22.21],
#         'humid': [23.83],
#         'co2': [710],
#         'voc': [139],
#         'pm25': [5]
#     }

#     # The x positions for the bars
#     params = list(predicted_data.keys())  # Use keys from predicted_data
#     num_params = len(params)
#     width = 0.35  # The width of the bars

#     fig, ax = plt.subplots(figsize=(10, 6))

#     # Add bars for predicted data (red)
#     for i, (param, values) in enumerate(predicted_data.items()):
#         ax.bar(i - width/2, values, width, label=f'Predicted {param}', color='orange')

#     # Add bars for live data (blue)
#     for i, (param, values) in enumerate(live_data.items()):
#         ax.bar(i + width/2, values, width, label=f'Live {param}', color='green')

#     # Add titles, labels, legend, and grid
#     ax.set_ylabel('Values')
#     ax.set_title('Predicted vs Live Data')

#     # Set the x-tick labels to only show each parameter name (no "Predicted" or "Live")
#     ax.set_xticks(range(num_params))  # Set tick positions for each parameter
#     ax.set_xticklabels(params)  # Only use the parameter names as labels

#     ax.legend(loc='upper left', ncol=2)
#     ax.set_ylim(0, 750)  # Set y-limit to make the bars visible

#     # Show the chart
#     plt.tight_layout()

#     # Save the chart as a PNG image
#     plt.savefig('predicted_vs_live_data.png')
#     print("Plot saved as predicted_vs_live_data.png")

#     plt.show()



# Example usage of prediction function
# print(temp_test_prediction())
# print(humid_test_prediction())
# print(co2_test_prediction())
# print(voc_test_prediction())
# print(pm25_test_prediction())
# print(f"Predicted Temperature: {predicted_data}")

# temp_test_prediction()
# humid_test_prediction()
# co2_test_prediction()
# voc_test_prediction()
# pm25_test_prediction()

# compare_live_predected()
# print(live_sensor_data())
# predicted_values = predicted_data
# pred_temp_value = predicted_values[1]
# pred_temp_humid = predicted_values[3]
# print(pred_temp_value)
# print(pred_temp_humid)
# print(predicted_data)
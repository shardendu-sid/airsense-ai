from src.sensor_api import sensor_api_connection
from src.models import model_test_with_live_data
from src import train
from src.air_cooler_integration import air_cooler
from src.WiFi_Socket import tapo_info
from src.current_weather_outdoor import current_weather_outside
from datetime import datetime

import time
import psycopg2
import csv
import os
import pandas as pd

from tabulate import tabulate

import mlflow
import mlflow.sklearn
import datetime


host = os.environ.get('CONTAINER_IP')
port = os.environ.get('PORT')
database = os.environ.get('DATABASE')
user = os.environ.get('USER')
password = os.environ.get('PASS_WORD')

predicted_csv = os.environ.get('predicted_data')


traind_models = "/Users/shardendujha/thesis-project-final-data/traind_models_score/trained_models.csv"


def outside_weather_now():
    while True:
        current_weather_outside.outdoor_weather()
        time.sleep(300)

    
def awair_row_data():
    while True:

        sensor_api_connection.awair_api_call()
        time.sleep(300)
        
def train_model():
    while True:
        train.temp_pred_model()
        train.humid_pred_model()
        train.co2_pred_model()
        train.voc_pred_model()
        train.pm25_pred_model()

        model_scores = train.trained_models
        
        current_date = datetime.date.today()
        new_dict = {"date": current_date}
        new_dict.update(model_scores)
        new_dict['date'] = new_dict['date'].isoformat()

        csv_file = "/Users/shardendujha/thesis-project-final-data/traind_models_score/trained_models.csv"
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=new_dict.keys())
            if f.tell() == 0:  # If the file is empty, write header
                writer.writeheader()  

            writer.writerow(new_dict)  

    

        time.sleep(86400) #86400/24hrs



def model_execution_with_live_data():

    while True:

        # mlflow.set_tracking_uri("http://127.0.0.1:5002")
        

        model_test_with_live_data.temp_test_prediction()
        
        model_test_with_live_data.humid_test_prediction()
        
        model_test_with_live_data.co2_test_prediction()
        
        model_test_with_live_data.voc_test_prediction()
        
        model_test_with_live_data.pm25_test_prediction()
        
        predicted_values = model_test_with_live_data.predicted_data
        print(predicted_values)

        pred_temp_value = predicted_values[1]
        pred_temp_humid = predicted_values[3]
        
        pred_temp_value_only = []
        for k,temp in pred_temp_value.items():
            pred_temp_value_only.append(temp)

        pred_humid_value_only = []

        for k,humid in pred_temp_humid.items():
            pred_humid_value_only.append(humid)

        print(temp,humid)
       

        air_cooler.air_coller_integration(temp,humid)
        
        # CSV file writing
        merged_data = {}
        for item in predicted_values:
            merged_data.update(item)

        # Check if the file exists and write data accordingly
        with open(predicted_csv, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=merged_data.keys())
            if csvfile.tell() == 0:  # If the file is empty, write header
                writer.writeheader()
            writer.writerow(merged_data)
        
        
        time.sleep(900)
        
        
     
def energy_consumption():
    while True:
        tapo_info.energy_time_calculation()
        # time.sleep(300)


    
if __name__ == '__main__':
    
    
    import concurrent.futures
    import warnings

    def suppress_warnings():
        warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')

    # Call suppress_warnings() before executing the functions that trigger the warnings
    suppress_warnings()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit both functions for execution as before
        current_weather = executor.submit(outside_weather_now)
        api_row_data = executor.submit(awair_row_data)
        train_model_twenty_foue_hrs = executor.submit(train_model)
        # future_data = executor.submit(data_preprocess)
        future_model = executor.submit(model_execution_with_live_data)
        energy_data = executor.submit(energy_consumption)

        # Wait for both functions to complete
        concurrent.futures.wait(
            [
                future_model,
                energy_data,
                api_row_data,
                train_model_twenty_foue_hrs,
                current_weather,
            ]
        )

       
       
    #    api_row_data,future_data

    








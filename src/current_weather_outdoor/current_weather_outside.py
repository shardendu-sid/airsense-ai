import requests
import os
import csv
import psycopg2
import datetime
import pytz

host = os.environ.get('DB_HOST')
port = os.environ.get('DB_PORT')
database = os.environ.get('DB_NAME')
user = os.environ.get('DB_USER')
password = os.environ.get('DB_PASSWORD')
url = os.environ.get('API_URL')

weather_apikey = os.environ.get('weather_api_key')
weathercsv_data = os.environ.get('weather_csv_data')

def outdoor_weather():
    # weather_data = []
    while True:

        # Replace 'YOUR_API_KEY' with your actual OpenWeatherMap API key
        api_key = weather_apikey

        # Replace 'New York' with the desired location for weather data
        location = 'Vantaa'

        # Define the API endpoint to get current weather data
        url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"

        # Make the API request to get the weather data
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()

           # Retrieve the current time and convert it to Helsinki timezone
            helsinki_timezone = pytz.timezone('Europe/Helsinki')
            current_time_local = datetime.datetime.now().astimezone(helsinki_timezone)
            formatted_time = current_time_local.strftime("%Y-%m-%d %H:%M:%S")

            # Access the required fields from the data
            weather_main = data['weather'][0].get('main', '-')
            main_temp = data['main'].get('temp', '-')
            main_feels_like = data['main'].get('feels_like', '-')
            main_pressure = data['main'].get('pressure', '-')
            main_humidity = data['main'].get('humidity', '-')
            visibility = data.get('visibility', '-')
            wind_speed = data['wind'].get('speed', '-')
            # snowfall = data.get('snow', {}).get('1h', '-')
            city_name = data.get('name', '-')
            country_code = data['sys'].get('country', '-')

            # print(data)
        

            headers = ['Timestamp','Temperature', 'Feels like', 'Pressure', 'Humidity',
                    'Visibility', 'Wind Speed', 'City Name', 'Country Code']

            zip_data = [
                (
                    formatted_time,main_temp, main_feels_like, main_pressure, main_humidity,
                    visibility, wind_speed, city_name, country_code
                )
            ]

            weather_data = []
            result_dict = dict(zip(headers, zip_data[0]))
            weather_data.append(result_dict)
           
            # print(weather_data)
            # print('')
            # print('                              Current Weather Outside')
            # print(tabulate(zip_data, headers, tablefmt='pretty'))

            column = ['Timestamp','Temperature', 'Feels like', 'Pressure', 'Humidity',
                    'Visibility', 'Wind Speed', 'City Name', 'Country Code']
            csv_file = weathercsv_data
            path = weathercsv_data

            if os.path.exists(path):
                with open(csv_file, "a+") as add_file:
                    writer = csv.DictWriter(add_file, fieldnames=column)
                    

                    for data in weather_data:
                        writer.writerow(data)

            else:
                try:
                    with open(csv_file, "w") as write_file:
                        writer = csv.DictWriter(write_file, fieldnames=column)
                        writer.writeheader()

                        for data in weather_data:
                            writer.writerow(data)
                except ValueError:
                    print("I/O error")

            conn1 = psycopg2.connect(
                host = host,
                port = port,
                user = user,
                database = database,
                password = password
            )
            conn1.autocommit=True
            cur1 = conn1.cursor()
            cur1.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = 'awair'")
            exists = cur1.fetchone()

            if not exists:
                cur1.execute("CREATE DATABASE awair")
            conn1.set_session(autocommit=True)

            try:
                conn = psycopg2.connect(
                    host=host,
                    port=port,
                    user=user,
                    database=database,
                    password=password
                )

            except psycopg2.Error as e:
                print(e)

            try:
                cur =conn.cursor()
            except psycopg2.Error as e:
                print("Error: Could not get the crusor to the database")
                print(e)

            conn.set_session(autocommit=True)

            try:
                cur.execute("CREATE TABLE IF NOT EXISTS current_weather(id BIGSERIAL PRIMARY KEY, \
                            Timestamp timestamp,Temperature NUMERIC,\
                            Feelslike NUMERIC, Pressure NUMERIC, Humidity NUMERIC,\
                            Visibility NUMERIC, WindSpeed NUMERIC, CityName VARCHAR,\
                             CountryCode VARCHAR, CONSTRAINT unique_current_weather UNIQUE (Timestamp)) ")
                
            except psycopg2.Error as e:
                print("Error: Issue creating table")
                print(e)
            
            

            sql = "INSERT INTO current_weather(Timestamp,Temperature, Feelslike, Pressure, Humidity, \
                Visibility, WindSpeed,CityName,CountryCode) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) \
                ON CONFLICT DO NOTHING;"

            try:
                for data in weather_data:
                    
                    values = (
                        data['Timestamp'],
                        data['Temperature'],
                        data['Feels like'],
                        data['Pressure'],
                        data['Humidity'],
                        data['Visibility'],
                        data['Wind Speed'],
                        data['City Name'],
                        data['Country Code']
                    )
                    cur.execute(sql, values)

            except psycopg2.Error as e:
                print("Error: Issue inserting data")
                print(e)
                
            
            
            return weather_data
        
     

   
       
        

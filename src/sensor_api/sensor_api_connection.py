# imported all the necessaries files that we needed

import requests
import json
import csv
import psycopg2
import os
import pytz
from dateutil import parser
import time

# avoide system sleep type in command terminal "caffeinate" press enter
# to stop caffeinate press "Ctrl+C"
# Created a function in which connection is established

host = os.environ.get("DB_HOST")
port = os.environ.get("DB_PORT")
database = os.environ.get("DB_NAME")
user = os.environ.get("DB_USER")
password = os.environ.get("DB_PASSWORD")

url = os.environ.get("API_URL")


def awair_api_call():

    conn1 = psycopg2.connect(
        host=host, port=port, database=database, user=user, password=password
    )

    conn1.autocommit = True

    cur1 = conn1.cursor()
    cur1.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = 'awair'")
    exists = cur1.fetchone()

    if not exists:
        cur1.execute("CREATE DATABASE awair")

    conn1.set_session(autocommit=True)

    try:
        conn = psycopg2.connect(
            host=host, port=port, database=database, user=user, password=password
        )

    except psycopg2.Error as e:
        print(e)

    try:
        cur = conn.cursor()
    except psycopg2.Error as e:
        print("Error: Could not get the crusor to the database")
        print(e)

    conn.set_session(autocommit=True)

    try:

        cur.execute(
            "CREATE TABLE IF NOT EXISTS awair_data(awair_id BIGSERIAL PRIMARY KEY, \
                                                timestamp timestamp,score int, \
                                                dew_point NUMERIC,temp NUMERIC,humid NUMERIC, \
                                                abs_humid NUMERIC,co2 int,co2_est int, \
                                                co2_est_baseline int, \
                                                voc int,voc_baseline int,voc_h2_raw int, \
                                                voc_ethanol_raw int,pm25 int,pm10_est int,location VARCHAR);"
        )
    except psycopg2.Error as e:
        print("Error: Issue creating table")
        print(e)

    try:
        cur.execute(
            "CREATE TABLE IF NOT EXISTS awair_owner_details(owner_id SERIAL PRIMARY KEY, \
                    owner_name VARCHAR(50), \
                    owner_country VARCHAR(50), owner_address VARCHAR(50), owner_phone VARCHAR, owner_email VARCHAR);"
        )
    except psycopg2.Error as e:
        print("Error: Issue creating table")
        print(e)

    cur.execute(
        "INSERT INTO awair_owner_details(owner_name,owner_country,owner_address,owner_phone,owner_email) VALUES ('Shardendu Jha', \
                    'Finland','Janonhanta 1 C 200','0442138793','apps00@gmail.com')"
    )

    data1 = None
    while True:
        List2 = []

        Url = url  # api url path

        request1 = requests.request("GET", Url, timeout=30)
        data1 = request1.json()

        # Parse the timestamp using dateutil.parser
        utc_timestamp = parser.parse(data1["timestamp"])

        # Convert to Helsinki time
        helsinki_timezone = pytz.timezone("Europe/Helsinki")
        helsinki_timestamp = utc_timestamp.astimezone(helsinki_timezone)

        # Format the timestamp to display only date and time
        data1["timestamp"] = helsinki_timestamp.strftime("%Y-%m-%d %H:%M:%S")

        add_new_col = {"location": "Janonhanta1,Vantaa,Finland"}
        add_bew_col_serial = {}
        data1.update(add_bew_col_serial)
        data1.update(add_new_col)

        List2.append(data1)

        headers = [
            "timestamp",
            "score",
            "temp",
            "humid",
            "co2",
            "voc",
            "pm25",
            "pm10_est",
            "location",
        ]
        table_data = []
        for row in List2:
            table_data.append([row.get(key, "") for key in headers])
        # print('')
        # print('                                     Live Sensor Parameters')
        # print(tabulate(table_data, headers=headers, tablefmt='pretty'))
        # print(List2)
        # time.sleep(300)# call every 5 min

        values = [v for k, v in List2[0].items()]
        sql = "INSERT INTO awair_data (timestamp,score,dew_point,temp,humid,abs_humid, \
                        co2,co2_est,co2_est_baseline,voc,voc_baseline,voc_h2_raw, \
                        voc_ethanol_raw,pm25,pm10_est,location) VALUES ({})".format(
            ",".join(["%s"] * len(values))
        )
        cur.execute(sql, values)

        # save data as csv file

        c_columns = [
            "timestamp",
            "score",
            "dew_point",
            "temp",
            "humid",
            "abs_humid",
            "co2",
            "co2_est",
            "co2_est_baseline",
            "voc",
            "voc_baseline",
            "voc_h2_raw",
            "voc_ethanol_raw",
            "pm25",
            "pm10_est",
            "location",
        ]

        csv_file = os.environ.get("CSV_FILE")
        path = os.environ.get("CSV_PATH")

        if os.path.exists(path):
            with open(csv_file, "a+") as add_obj:
                writer = csv.DictWriter(add_obj, fieldnames=c_columns)

                for data in List2:
                    writer.writerow(data)
        else:
            try:
                with open(csv_file, "w") as awair_file:
                    writer = csv.DictWriter(awair_file, fieldnames=c_columns)
                    writer.writeheader()

                    for data in List2:
                        writer.writerow(data)

            except ValueError:
                print("I/O error")

        # save data as json file.
        json_file = os.environ.get("JSON_FILE")
        path = os.environ.get("JSON_PATH")

        if os.path.exists(path):
            with open(json_file, "r") as s_file:
                try:
                    existing_records = json.load(s_file)
                except json.decoder.JSONDecodeError:
                    existing_records = []

            for record in List2:
                existing_records.append(record)

            with open(json_file, "w") as j_file:
                json.dump(existing_records, j_file, sort_keys=True, indent=4)

        else:
            with open(json_file, "w") as f:
                json.dump(List2, f, sort_keys=True, indent=4)

        # print(List2)
        return List2
        # print(List2)
        # time.sleep(300)# call every 5 min


# if __name__ == '__main__':

#     awair_api_call()
#     time.sleep(300)
    

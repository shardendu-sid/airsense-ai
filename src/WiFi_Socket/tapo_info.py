from datetime import datetime
import requests
import time
import os
import csv 
import psycopg2

host = os.environ.get("CONTAINER_IP")
port = os.environ.get("PORT")
database = os.environ.get("DATABASE")
user = os.environ.get("USER")
password = os.environ.get("PASS_WORD")

smartthings_url = os.environ.get('smart_things_url')
# deviceendpoint_smartthings = os.environ.get('device_endpoint_smartthings')
accesstoken_smarttings = os.environ.get('access_token_smartthings')
deviceid = os.environ.get('device_id')
energy_cal_starttime = os.environ.get('energy_cal_start_time')
energy_cal_endtime = os.environ.get('energy_cal_end_time')
electricity_cal_cost = os.environ.get('electricity_cost')

def check_device_status(access_token, device_id):
    api_url = smartthings_url
    status_endpoint = f'/v1/devices/{device_id}/status'

    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }

    # Make a GET request to retrieve device status
    response = requests.get(api_url + status_endpoint, headers=headers)

    if response.status_code == 200:
        device_status = response.json()
        return device_status
    else:
        print(f"Failed to get device status. Status code: {response.status_code}")
        return None
    
def calculate_total_used_time(time_data):
    start_times = []
    end_times = []
    
    for entry in time_data:
        if 'start_time' in entry:
            start_times.append(datetime.strptime(entry['start_time'], '%Y-%m-%dT%H:%M:%S.%fZ'))
        elif 'end_time' in entry:
            end_times.append(datetime.strptime(entry['end_time'], '%Y-%m-%dT%H:%M:%S.%fZ'))

    total_used_time = 0
    paired_times = zip(sorted(start_times), sorted(end_times))
    for start_time, end_time in paired_times:
        total_used_time += (end_time - start_time).total_seconds()
        
    return total_used_time


def energy_time_calculation():
    while True:

        # Usage: Call the function with your access token and device ID to check the status
        access_token = accesstoken_smarttings
        device_id = deviceid

        status = check_device_status(access_token, device_id)
        if status is not None:
            pass
            # print("Device status:", status)

        # Accessing switch status value
        switch_status = status['components']['main']['switch']['switch']['value']
        switch_timestamp = status['components']['main']['switch']['switch']['timestamp']
        # Printing the switch status
        # print("Switch status:", switch_status)
        # print("Switch timestamp:", switch_timestamp)

        s_columns = ['start_time']
        e_columns = ['end_time']
        start_times = []
        end_times = []
        if switch_status == "on":
            start_on_timestamp = switch_timestamp
            start_times.append(start_on_timestamp)
            csv_file = energy_cal_starttime
            path = energy_cal_starttime
            

            if os.path.exists(path):
                with open(csv_file, "a+") as add_obj:
                    writer = csv.DictWriter(add_obj, fieldnames=s_columns)
                    file_content = open(csv_file).read()
                    for data in start_times:
                        if data not in file_content:
                            writer.writerow({'start_time': data})
            else:
                try:
                    with open(csv_file, "w") as awair_file:
                        writer = csv.DictWriter(awair_file, fieldnames=s_columns)
                        writer.writeheader()
                        
                        for data in start_times:
                            writer.writerow({'start_time': data})
                            
                except ValueError:
                    print("I/O error")

            # print(start_times)

            conn1 = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password
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
                    database=database,
                    user=user,
                    password=password
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
            
                cur.execute("CREATE TABLE IF NOT EXISTS start_time_data(id BIGSERIAL PRIMARY KEY,start_time timestamp,\
                        CONSTRAINT unique_start_time UNIQUE (start_time));")
            except psycopg2.Error as e:
                print("Error: Issue creating table")
                print(e)

                
            for timestamp_value in start_times:
                sql = "INSERT INTO start_time_data(start_time) VALUES (%s) ON CONFLICT DO NOTHING;"

                cur.execute(sql, (timestamp_value,))


            
            
        elif switch_status == "off":
            end_of_timestamp = switch_timestamp
            end_times.append(end_of_timestamp)
            csv_file = energy_cal_endtime
            path = energy_cal_endtime
            

            if os.path.exists(path):
                with open(csv_file, "a+") as add_obj:
                    writer = csv.DictWriter(add_obj, fieldnames=e_columns)

                    file_content = open(csv_file).read()
                    
                    for data in end_times:
                        if data not in file_content:
                            writer.writerow({'end_time': data})
            else:
                try:
                    with open(csv_file, "w") as awair_file:
                        writer = csv.DictWriter(awair_file, fieldnames=e_columns)
                        writer.writeheader()
                        
                        for data in end_times:
                            writer.writerow({'end_time': data})
                            
                except ValueError:
                    print("I/O error")
        # else:
        #     print("Invalid timestamp")
       
        conn1 = psycopg2.connect(
                    host=host,
                    port=port,
                    database=database,
                    user=user,
                    password=password
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
                database=database,
                user=user,
                password=password
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
        
            cur.execute("CREATE TABLE IF NOT EXISTS end_time_data(id BIGSERIAL PRIMARY KEY,end_time timestamp,\
                        CONSTRAINT unique_end_time UNIQUE (end_time));")
        except psycopg2.Error as e:
            print("Error: Issue creating table")
            print(e)

            
        for timestamp_value in end_times:
            sql = "INSERT INTO end_time_data (end_time) VALUES (%s) ON CONFLICT DO NOTHING;"
            cur.execute(sql, (timestamp_value,))

        
        
        csv_file = energy_cal_endtime
        path = energy_cal_endtime

        merged_data = []

        if os.path.exists(path):
            with open(csv_file, 'r') as readfile:
                reader = csv.DictReader(readfile)

                for row in reader:
                    merged_data.append(row)

        csv_file = energy_cal_starttime
        path = energy_cal_starttime

        if os.path.exists(path):
            with open(csv_file, 'r') as readfile:
                reader = csv.DictReader(readfile)

                for row in reader:
                    merged_data.append(row)

        # print(merged_data)
        # total_used_time = calculate_total_used_time(merged_data)
        # print(f"The total used time is: {total_used_time} seconds")
        start_time_value = []
        end_time_vlaue = []
        for data in merged_data:
            for key, value in data.items():
                if key == 'start_time':
                    start_time_value.append(value)
                    
                if key == 'end_time':
                    end_time_vlaue.append(value)      
        # print(start_time_value)
        # print(end_time_vlaue)
        # time.sleep(300)
        electricity_cost_data = []

        for start, end in zip(start_time_value, end_time_vlaue):
            starttime = datetime.fromisoformat(start[:-1]) 
            endtime = datetime.fromisoformat(end[:-1])
            duration = (endtime - starttime).total_seconds() / 60
            # print(f"Start time: {starttime}, End time: {endtime}, Used time: {duration} minutes")
            # time.sleep(2)

            device_capacity = 65 # in watts, device_consume_electricity
            min_in_hrs= round(duration/60,2) #calculate_total_min_in_hours_in_day
            watt_hours = round(device_capacity * min_in_hrs,2) # watt_hours_per_day
            kwh = round(watt_hours / 1000,2) # total_kwh_per_day
            price_per_kwh = 0.13 # 0.13 cents per hour,price_per_kwh
            e_tranf_cost = round(0.3 * kwh,2) # electricity_transfer_cost
            e_tax_cost = round(0.3 * kwh,2) # electricity_tax
            calculate_electricity_cost= price_per_kwh * kwh
            total_cost = round(calculate_electricity_cost + e_tranf_cost + e_tax_cost,2) # total_eletricity_cost
            electricity_cost_data = [{'Start_time': start,'End_time': end,
                                     'Used_time_hrs': min_in_hrs,
                                     'Price_per_kwh_cents': price_per_kwh,
                                     'Transfer_per_kwh': e_tranf_cost,
                                     'Tax_per_kwh': e_tax_cost,
                                     'Total_cost_euro': total_cost}]
            
            column_name  = ["Start_time", "End_time", "Used_time_hrs", "Price_per_kwh_cents",
                            "Transfer_per_kwh", "Tax_per_kwh", "Total_cost_euro"]
            
            csv_file = electricity_cal_cost
            path = electricity_cal_cost

            if os.path.exists(path):
                with open(csv_file, 'a+') as addfile:
                    writer = csv.DictWriter(addfile, fieldnames=column_name)

                    file_content = open(csv_file).read()
                    
                    for data in electricity_cost_data:
                        if f"{data['Start_time']},{data['End_time']}" not in file_content:
                            writer.writerow(data)

            else:
                try:
                    with open(csv_file, 'w') as writefile:
                        writer = csv.DictWriter(writefile, fieldnames=column_name)
                        writer.writeheader()
                        
                        
                        for data in electricity_cost_data:
                            writer.writerow(data)

                except ValueError:
                    print("I/O error")


            conn1 = psycopg2.connect(
                    host=host,
                    port=port,
                    database=database,
                    user=user,
                    password=password
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
                    database=database,
                    user=user,
                    password=password
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
            
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS cost_analysis_data (
                        id BIGSERIAL PRIMARY KEY,
                        start_time TIMESTAMP,
                        end_time TIMESTAMP,
                        used_time_hrs NUMERIC,
                        price_per_kwh_cents NUMERIC,
                        transfer_per_kwh NUMERIC,
                        tax_per_kwh NUMERIC,
                        total_cost_euro NUMERIC,
                        CONSTRAINT unique_cost_analysis_data UNIQUE (id)
                    );
                """)
            except psycopg2.Error as e:
                print("Error: Issue creating table")
                print(e)

               
            
            check_sql = """
                SELECT COUNT(*) FROM cost_analysis_data
                WHERE Start_time = %s AND End_time = %s
            """

            # Insert data into the PostgreSQL database
            for data in electricity_cost_data:
                # Check if data exists in the table
                cur.execute(check_sql, (data['Start_time'], data['End_time']))
                result = cur.fetchone()[0]
                
                # If data doesn't exist, insert it
                if result == 0:
                    sql = """
                        INSERT INTO cost_analysis_data (
                            Start_time, End_time, Used_time_hrs, Price_per_kwh_cents,
                            Transfer_per_kwh, Tax_per_kwh, Total_cost_euro
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING;
                    """

                    values = (
                        data['Start_time'],
                        data['End_time'],
                        data['Used_time_hrs'],
                        data['Price_per_kwh_cents'],
                        data['Transfer_per_kwh'],
                        data['Tax_per_kwh'],
                        data['Total_cost_euro']
                    )

                    cur.execute(sql, values)

                
            


        #     print(electricity_cost_data)
        #     print(f"Start time: {starttime}, End time: {endtime}, Used time: {duration} minutes")
        
#         time.sleep(300)
        

# energy_time_calculation()
# time.sleep(300)

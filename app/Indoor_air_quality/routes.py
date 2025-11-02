from flask import render_template, request
from app.Indoor_air_quality import main
from seperate import process_device_info
from flask_login import login_required
import warnings
import os
from flask import jsonify
import pandas as pd
from src.WiFi_Socket import tapo_socket
from src.models.evaluate_models import (
    perform_linear_regression,
    perform_random_forest,
    perform_support_vector_machine,
    perform_k_nearest_neighbors,
    perform_decision_trees,
)
from notebook.exploratory_data_analysis import (
    generate_base64_plot,
    generate_correlation_heatmap,
)
import pytz


predicted_csv = os.environ.get('predicted_data')
weather_csv_data = os.environ.get('weather_csv_data')
CSV_FILE = os.environ.get('CSV_FILE')
energy_cal_start_time = os.environ.get('energy_cal_start_time')
electricity_cost = os.environ.get('electricity_cost')
processed_data = os.environ.get('load_processed_data')

awair_data_one = os.environ.get("awair_csv_one")
awair_data_two = os.environ.get("awair_csv_two")
trained_model_data = os.environ.get("trained_models")

def suppress_warnings(): 
    warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')

@main.route('/dashboard')
@login_required
def display_dashboard():
    
    device_id = os.environ.get('device_id')
    access_token = os.environ.get('access_token_smartthings')
    is_on = tapo_socket.device_status(access_token, device_id)
   
    df_w = pd.read_csv(weather_csv_data, header=0)
    last_line = df_w.tail(1)
    weather_data = last_line.values

    df_indor = pd.read_csv(CSV_FILE, header=0)
    last_line_indoor = df_indor.tail(1)
    sensor_data = last_line_indoor.values

    df_predicted = pd.read_csv(predicted_csv, header=0)
    last_line_prediction = df_predicted.tail(1)
    predicted_data = last_line_prediction.values

    return render_template('home.html',weather_data=weather_data,sensor_data=sensor_data,predicted_data=predicted_data,air_cooler_status=is_on)

@main.route('/dashboard/status')
@login_required
def display_device_info():
    
    
    device_id = os.environ.get('device_id')
    access_token = os.environ.get('access_token_smartthings')
    is_on = tapo_socket.device_status(access_token, device_id)

    device_info, switch_status, switch_timestamp = process_device_info()
    return render_template('device_info.html', device_info=device_info, switch_status=switch_status, switch_timestamp=switch_timestamp,air_cooler_status=is_on,)

@main.app_errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

@main.route('/latest-data')
@login_required
def latest_data():
    try:
        
        df = pd.read_csv(predicted_csv)  
        df = df.tail(100)  
        return jsonify(df.to_dict(orient='records'))
    except FileNotFoundError:
        return jsonify([]) 


@main.route('/weather-data')
@login_required
def weather_data():
    try:
        
        df = pd.read_csv(weather_csv_data)  
        df = df.tail(100)  
        return jsonify(df.to_dict(orient='records'))
    except FileNotFoundError:
        return jsonify([])  


@main.route('/energy-data')
@login_required
def energy_data():
    try:
        
        df = pd.read_csv(electricity_cost) 
        df = df.tail(100)  
        return jsonify(df.to_dict(orient='records'))
    except FileNotFoundError:
        return jsonify([])  

@main.route('/sensor-data')
@login_required
def sensor_data():
    try:
        
        df = pd.read_csv(CSV_FILE,low_memory=False)  
        df = df.tail(100)  
        return jsonify(df.to_dict(orient='records'))
    except FileNotFoundError:
        return jsonify([])  

@main.route('/display-sensor-data')
@login_required
def display_sensor_data():

    device_id = os.environ.get('device_id')
    access_token = os.environ.get('access_token_smartthings')
    is_on = tapo_socket.device_status(access_token, device_id)
    
    search_query = request.args.get('search', '')  
    page = request.args.get('page', 1, type=int)
    per_page = 20

    data = pd.read_csv(CSV_FILE,low_memory=False)

    if search_query:
      
        data['timestamp'] = data['timestamp'].astype(str)

  
        data = data[data['timestamp'].str.contains(search_query, case=False, na=False)]

    total_rows = len(data)
    total_pages = (total_rows - 1) // per_page + 1

    data_subset = data.iloc[(page - 1) * per_page : page * per_page]
    data_dicts = data_subset.to_dict(orient='records')
    return render_template('awair_sensor.html', table=data_dicts, total_pages=total_pages, current_page=page, air_cooler_status=is_on)

@main.route('/display-predicted-data')
@login_required
def display_predicted_data():

    device_id = os.environ.get('device_id')
    access_token = os.environ.get('access_token_smartthings')
    is_on = tapo_socket.device_status(access_token, device_id)
    
    search_query = request.args.get('search', '')  
    page = request.args.get('page', 1, type=int)
    per_page = 20

    data = pd.read_csv(predicted_csv)

    if 'Temp_Pred' in data.columns:
        data['Temp_Pred'] = data['Temp_Pred'].round(2)
    if 'Humid_Pred' in data.columns:
        data['Humid_Pred'] = data['Humid_Pred'].round(2)
    if 'Tempe_Test_S' in data.columns:
        data['Tempe_Test_S'] = data['Tempe_Test_S'].round(2)
    if 'Humid_Test_S' in data.columns:
        data['Humid_Test_S'] = data['Humid_Test_S'].round(2)
    if 'Co2_Test_S' in data.columns:
        data['Co2_Test_S'] = data['Co2_Test_S'].round(2)
    if 'VOC_Test_S' in data.columns:
        data['VOC_Test_S'] = data['VOC_Test_S'].round(2)
    if 'Pm25_Test_S' in data.columns:
        data['Pm25_Test_S'] = data['Pm25_Test_S'].round(2)

    if search_query:
        
        data = data[data['DateTime'].str.contains(search_query, case=False)]

    total_rows = len(data)
    total_pages = (total_rows - 1) // per_page + 1

    data_subset = data.iloc[(page - 1) * per_page : page * per_page]
    data_dicts = data_subset.to_dict(orient='records')
    return render_template('ml_prediction.html', table=data_dicts, total_pages=total_pages, current_page=page, air_cooler_status=is_on)

@main.route('/display-energy-data')
@login_required
def display_energy_data():

    device_id = os.environ.get('device_id')
    access_token = os.environ.get('access_token_smartthings')
    is_on = tapo_socket.device_status(access_token, device_id)
    
    search_query = request.args.get('search', '') 
    page = request.args.get('page', 1, type=int)
    per_page = 5

    data = pd.read_csv(electricity_cost)

    local_tz = pytz.timezone('Europe/Helsinki')  

    
    data['Start_time'] = pd.to_datetime(data['Start_time']).dt.tz_convert(local_tz).dt.strftime('%Y-%m-%d %H:%M:%S')
    data['End_time'] = pd.to_datetime(data['End_time']).dt.tz_convert(local_tz).dt.strftime('%Y-%m-%d %H:%M:%S')

    data['combined_time'] = data['Start_time'].astype(str) + '-' + data['End_time'].astype(str)

    

    if search_query:
        
        data = data[data['combined_time'].str.contains(search_query, case=False, na=False)]

    total_rows = len(data)
    total_pages = (total_rows - 1) // per_page + 1

    data_subset = data.iloc[(page - 1) * per_page : page * per_page]
    data_dicts = data_subset.to_dict(orient='records')
    return render_template('energy_cost.html', table=data_dicts, total_pages=total_pages, current_page=page, air_cooler_status=is_on)

@main.route('/display-weather-data')
@login_required
def display_weather_data():

    device_id = os.environ.get('device_id')
    access_token = os.environ.get('access_token_smartthings')
    is_on = tapo_socket.device_status(access_token, device_id)
    
    search_query = request.args.get('search', '')  # Get the search query
    page = request.args.get('page', 1, type=int)
    per_page = 20

    data = pd.read_csv(weather_csv_data)

    
    
    if search_query:
      
        data = data[data['Timestamp'].str.contains(search_query, case=False)]

    total_rows = len(data)
    total_pages = (total_rows - 1) // per_page + 1

    data_subset = data.iloc[(page - 1) * per_page : page * per_page]
    data_dicts = data_subset.to_dict(orient='records')
    return render_template('weather.html', table=data_dicts, total_pages=total_pages, current_page=page, air_cooler_status=is_on)


@main.route('/display-about-project')
@login_required
def display_about_project():
    device_id = os.environ.get('device_id')
    access_token = os.environ.get('access_token_smartthings')
    is_on = tapo_socket.device_status(access_token, device_id)

    return render_template('about-project.html', air_cooler_status=is_on)


@main.route('/pick-model', methods=['GET', 'POST'])
@login_required
def pick_model():
    device_id = os.environ.get('device_id')
    access_token = os.environ.get('access_token_smartthings')
    is_on = tapo_socket.device_status(access_token, device_id)

    evaluation_result = None  
    
    if request.method == 'POST':
        selected_model = request.form.get('model')  
        
        if selected_model == 'linear-regression':
            
            evaluation_result = perform_linear_regression()
       
        elif selected_model == 'random-forest':
            evaluation_result = perform_random_forest()

        elif selected_model == 'support-vector-machine':
            
            evaluation_result = perform_support_vector_machine()
        elif selected_model == 'k-nearest-neighbors':
            
            evaluation_result = perform_k_nearest_neighbors()
        elif selected_model == 'decision-trees':
            evaluation_result = perform_decision_trees()
        else:
            
            return render_template('error.html', message='Invalid model selection')

    return render_template('pick-model.html', air_cooler_status=is_on, evaluation_result=evaluation_result)

@main.route('/data-analysis')
@login_required
def data_analysis():
    device_id = os.environ.get('device_id')
    access_token = os.environ.get('access_token_smartthings')
    is_on = tapo_socket.device_status(access_token, device_id)

    awair_csv_file = os.environ.get('CSV_FILE')
    awair_csv_data = pd.read_csv(awair_csv_file)
    
    
    selected_required_columns = awair_csv_data[['temp', 'humid', 'co2', 'voc', 'pm25']].copy()
    selected_required_columns.drop_duplicates(inplace=True)
    selected_required_columns.dropna(inplace=True)
   
    correlation_matrix = selected_required_columns.corr()

   
    correlation_heatmap = generate_correlation_heatmap(correlation_matrix)

    
    # Generate plots
    plots = {
    'temp': generate_base64_plot(selected_required_columns, 'temp', 'blue', 'Temperature Distribution', 'Temperature (°C)'),
    'humid': generate_base64_plot(selected_required_columns, 'humid', 'green', 'Humidity Distribution', 'Humidity'),
    'co2': generate_base64_plot(selected_required_columns, 'co2', 'pink', 'Co2 Distribution', 'Co2'),
    'voc': generate_base64_plot(selected_required_columns, 'voc', 'red', 'VOC Distribution', 'VOC'),
    'pm25': generate_base64_plot(selected_required_columns, 'pm25', 'yellow', 'Pm2.5 Distribution', 'Pm2.5'),

    }
    
    return render_template('exploratory_analysis.html', plots=plots, correlation_heatmap=correlation_heatmap, air_cooler_status=is_on)

@main.route('/system-design-tech')
@login_required
def system_design_tech():
    device_id = os.environ.get('device_id')
    access_token = os.environ.get('access_token_smartthings')
    is_on = tapo_socket.device_status(access_token, device_id)

    return render_template('system-design-tech.html', air_cooler_status=is_on)

@main.route('/contact')
@login_required
def contact():
    device_id = os.environ.get('device_id')
    access_token = os.environ.get('access_token_smartthings')
    is_on = tapo_socket.device_status(access_token, device_id)

    return render_template('contact.html', air_cooler_status=is_on)

@main.route('/add-new-device')
@login_required
def add_new_device():
    device_id = os.environ.get('device_id')
    access_token = os.environ.get('access_token_smartthings')
    is_on = tapo_socket.device_status(access_token, device_id)
    return render_template('add_new_device.html',air_cooler_status=is_on)


# comparative analysis
@main.route('/comparative-analysis')
@login_required
def comparative_analysis():
    device_id = os.environ.get('device_id')
    access_token = os.environ.get('access_token_smartthings')
    is_on = tapo_socket.device_status(access_token, device_id)

    awair_csv_data_01_03_2023 = awair_data_one
    
    selected_required_columns_one = pd.read_csv(awair_csv_data_01_03_2023, sep=",", quotechar='"')
    selected_required_columns_one.drop_duplicates(inplace=True)
    selected_required_columns_one.dropna(inplace=True)
    
    
    awair_csv_data_26_03_2023 = awair_data_two

    selected_required_columns_two = pd.read_csv(awair_csv_data_26_03_2023, sep=",", quotechar='"')
    selected_required_columns_two.drop_duplicates(inplace=True)
    selected_required_columns_two.dropna(inplace=True)
    
    # Generate plots
    plots_one = {
    'temp': generate_base64_plot(selected_required_columns_one, 'temp', 'blue', 'Temperature Distribution', 'Temperature (°C)'),
    'humid': generate_base64_plot(selected_required_columns_one, 'humid', 'green', 'Humidity Distribution', 'Humidity'),
    'co2': generate_base64_plot(selected_required_columns_one, 'co2', 'pink', 'Co2 Distribution', 'Co2'),
    
    }

    plots_two = {

    'temp': generate_base64_plot(selected_required_columns_two, 'temp', 'blue', 'Temperature Distribution', 'Temperature (°C)'),
    'humid': generate_base64_plot(selected_required_columns_two, 'humid', 'green', 'Humidity Distribution', 'Humidity'),
    'co2': generate_base64_plot(selected_required_columns_two, 'co2', 'pink', 'Co2 Distribution', 'Co2'),

    }
    
    return render_template('comparative_analysis.html', plots_one=plots_one,plots_two=plots_two, air_cooler_status=is_on)

# display train and test score
@main.route('/display-train-test-score')
@login_required
def display_train_test_score():

    device_id = os.environ.get('device_id')
    access_token = os.environ.get('access_token_smartthings')
    is_on = tapo_socket.device_status(access_token, device_id)
    
    search_query = request.args.get('search', '')  # Get the search query
    page = request.args.get('page', 1, type=int)
    per_page = 20

    data = pd.read_csv(trained_model_data)

    
    
    if search_query:
      
        data = data[data['date'].str.contains(search_query, case=False)]

    total_rows = len(data)
    total_pages = (total_rows - 1) // per_page + 1

    data_subset = data.iloc[(page - 1) * per_page : page * per_page]
    data_dicts = data_subset.to_dict(orient='records')
    return render_template('train_test_score.html', table=data_dicts, total_pages=total_pages, current_page=page, air_cooler_status=is_on)
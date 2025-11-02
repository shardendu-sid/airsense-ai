import datetime
import csv

data = {
    "temp_train_rmse": 0.02,
    "temp_test_rmse": 0.04,
    "temp_train_mae": 0.02,
    "temp_test_mae": 0.03,
    "temp_train_r2": 0.97,
    "temp_test_r2": 0.92,
    "humid_train_rmse": 0.86,
    "humid_test_rmse": 1.67,
    "humid_train_mae": 0.43,
    "humid_test_mae": 0.78,
    "humid_train_r2": 0.89,
    "humid_test_r2": 0.66,
    "co2_train_rmse": 59.73,
    "co2_test_rmse": 70.02,
    "co2_train_mae": 43.54,
    "co2_test_mae": 51.25,
    "co2_train_r2": 0.78,
    "co2_test_r2": 0.7,
    "voc_train_rmse": 120.19,
    "voc_test_rmse": 184.51,
    "voc_train_mae": 74.01,
    "voc_test_mae": 101.9,
    "voc_train_r2": 0.85,
    "voc_test_r2": 0.65,
    "pm25_train_rmse": 1.93,
    "pm25_test_rmse": 1.51,
    "pm25_train_mae": 0.58,
    "pm25_test_mae": 0.61,
    "pm25_train_r2": 0.91,
    "pm25_test_r2": 0.78,
}
current_date = datetime.date.today()
new_dict = {"date": current_date}
new_dict.update(data)
new_dict['date'] = new_dict['date'].isoformat()

csv_file = "/media/shardendujha/backup11/Home_Automation/mycsvfile.csv"
with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=new_dict.keys())
    writer.writeheader()  
    writer.writerow(new_dict)  

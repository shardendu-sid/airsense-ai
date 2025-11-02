from src.WiFi_Socket import tapo_socket
import os
import requests

def process_device_info():
    smartthings_url = os.environ.get('smart_things_url')
    accesstoken_smartthings = os.environ.get('access_token_smartthings')
    deviceid = os.environ.get('device_id')

    api_url = smartthings_url
    access_token = accesstoken_smartthings

    devices_endpoint = '/v1/devices'

    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }

    response = requests.get(api_url + devices_endpoint, headers=headers)
    device_info = []
    switch_status = None
    switch_timestamp = None

    if response.status_code == 200:
        devices = response.json().get('items', [])
        device_info.extend(devices)
        # for item in devices:
            # print(f"Label: {item.get('label')}")
            # print(f"Manufacturer Name: {item.get('manufacturerName')}")
            # print(f"Type: {item.get('type')}")
            # print(f"Create Time: {item.get('createTime')}")
            # print("\n")

            # viper_info = item.get('viper', {})
            # print(f"Viper Manufacturer: {viper_info.get('manufacturerName')}")
            # print(f"Model Name: {viper_info.get('modelName')}")
            # print(f"Software Version: {viper_info.get('swVersion')}")
            # print(f"Hardware Version: {viper_info.get('hwVersion')}")
            # print("\n")

    data_dict = tapo_socket.check_device_status(accesstoken_smartthings, deviceid)
    switch_status, switch_timestamp = extract_switch_status_and_timestamp(data_dict)

    return device_info, switch_status, switch_timestamp

def extract_switch_status_and_timestamp(d):
    value = d.get('components', {}).get('main', {}).get('switch', {}).get('switch', {}).get('value')
    timestamp = d.get('components', {}).get('main', {}).get('switch', {}).get('switch', {}).get('timestamp')
    return value, timestamp

# Usage example
# device_info, switch_status, switch_timestamp = process_device_info()
# print("Device Information:", device_info)
# print("Switch Status:", switch_status)
# print("Switch Timestamp:", switch_timestamp)

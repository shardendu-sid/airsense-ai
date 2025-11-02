import time
from WiFi_Socket import tapo_socket
from email_notifications import email_notification

prev_temperature = None  # Variable to store previous temperature value
prev_humidity = None

# Integrate with the wifi socket based on the predicted values
last_notification_time = 0
notification_interval = 300  # 5 minutes in seconds

def air_coller_integration(prediction_temperature, prediction_humid):
    
    global last_notification_time
    current_time = time.time()

    if (current_time - last_notification_time) >= notification_interval:
        if prediction_temperature >= 19 or prediction_humid <= 50:
            if not tapo_socket.is_air_cooler_power_on():
                tapo_socket.air_cooler_power_turn_on()
                email_notification.send_email_notification_turn_on()
                last_notification_time = current_time  # Update notification time

        else:
            if tapo_socket.is_air_cooler_power_on():
                tapo_socket.air_cooler_power_turn_off()
                email_notification.send_email_notification_turn_off()
                last_notification_time = current_time  # Update notification ti
           


    # time.sleep(1200)# call every 5 min 
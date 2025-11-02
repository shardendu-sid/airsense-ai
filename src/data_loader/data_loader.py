# Data loading and preprocessing

import pandas as pd
import os
current_directory = os.path.dirname(os.path.abspath(__file__))

def data_preprocessing():
    row_data = os.environ.get('CSV_FILE')
    processed_data = os.environ.get('load_processed_data')
    

    while True:
        try:
            # Load the collected data into a DataFrame
            awair_csv_data = pd.read_csv(row_data, low_memory=False)
            selected_required_columns = awair_csv_data[['temp', 'humid', 'co2', 'voc', 'pm25']].dropna().drop_duplicates()

            csv_file = processed_data
            if os.path.isfile(csv_file):
                # Load the existing data to check for duplicates
                existing_data = pd.read_csv(csv_file, low_memory=False)
                combined_data = pd.concat([existing_data, selected_required_columns]).drop_duplicates()
                combined_data.to_csv(csv_file, index=False)
            else:
                # Create a new CSV file if it doesn't exist
                selected_required_columns.to_csv(csv_file, header=True, index=False)

            # time.sleep(600)
            # time.sleep(30 * 24 * 60 * 60)  # Sleep for 30 days

        except Exception as e:
            print(f"Error occurred: {e}")

# if __name__ == '__main__':
#     data_preprocessing()



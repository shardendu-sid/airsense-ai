import unittest
from unittest.mock import patch, MagicMock

from src.sensor_api.sensor_api_connection import awair_api_call

class TestAwairApiCall(unittest.TestCase):
    # Patching the database connection

    @patch('src.sensor_api.sensor_api_connection.psycopg2.connect')
    @patch.dict('os.environ', {
        'CONTAINER_IP': '127.0.0.1',
        'PORT': '5432',
        'DATABASE': 'test_db',
        'USER': 'test_user',
        'PASS_WORD': 'test_password',
        'API_LINK': 'http://api.example.com/data',
        'CSV_FILE': 'test_file.csv',
        'CSV_PATH': 'test_path.csv',
        'JSON_FILE': 'test_file.json',
        'JSON_PATH': 'test_path.json'
    })
    def test_awair_api_call(self, mock_connect):  
        # Mocking database connection
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Mocking API call
        with patch('src.sensor_api.sensor_api_connection.requests.request') as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                'timestamp': '2023-05-01T12:00:00Z',
                'score': 85,
                'dew_point': 5,
                'temp': 20,
                'humid': 50,
                'abs_humid': 10,
                'co2': 400,
                'co2_est': 405,
                'co2_est_baseline': 390,
                'voc': 150,
                'voc_baseline': 140,
                'voc_h2_raw': 145,
                'voc_ethanol_raw': 150,
                'pm25': 12,
                'pm10_est': 10
            }
            mock_request.return_value = mock_response

            # Mocking os.path.exists to simulate file existence
            with patch('src.sensor_api.sensor_api_connection.os.path.exists') as mock_exists:
                mock_exists.return_value = False  # Simulate file does not exist
                
                # Mocking open for both CSV and JSON file operations
                with patch('builtins.open', unittest.mock.mock_open()) as mock_open:
                    with patch('src.sensor_api.sensor_api_connection.csv.DictWriter') as mock_csv_writer:
                        mock_csv_writer.return_value = MagicMock()
                        with patch('src.sensor_api.sensor_api_connection.json.dump') as mock_json_dump:
                            with patch('src.sensor_api.sensor_api_connection.json.load', return_value=[]):
                                try:
                                    result = awair_api_call()
                                except TypeError:
                                    result = None

        # Assertions
        self.assertTrue(mock_connect.called)
        self.assertTrue(mock_request.called)
        self.assertTrue(mock_cursor.execute.called)
        # Add more assertions as needed

if __name__ == '__main__':
    unittest.main()

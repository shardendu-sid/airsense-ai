
import unittest
from unittest.mock import patch, mock_open, MagicMock
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.train import temp_pred_model,humid_pred_model,co2_pred_model,voc_pred_model,pm25_pred_model

class TestTempPredModel(unittest.TestCase):

    @patch('os.environ.get', return_value='path_to_csv_file.csv')  
    @patch('pandas.read_csv', return_value=pd.DataFrame({'temp': [25, 26, 27], 'humid': [50, 60, 70], 'co2': [400, 500, 600], 'voc': [100, 150, 200], 'pm25': [10, 20, 30]}))  
    @patch('sklearn.model_selection.train_test_split', return_value=(pd.DataFrame({'humid': [50], 'co2': [500], 'voc': [150], 'pm25': [20]}), pd.DataFrame({'humid': [60], 'co2': [600], 'voc': [200], 'pm25': [30]}), pd.Series([25]), pd.Series([26])))  
    @patch('sklearn.ensemble.RandomForestRegressor')  
    @patch('sklearn.metrics.mean_squared_error', return_value=0.7)  
    @patch('sklearn.metrics.mean_absolute_error', return_value=0.37)  
    @patch('sklearn.metrics.r2_score', return_value=float('nan'))  
    
    def test_temp_pred_model(self, mock_r2_score, mock_mae, mock_mse, mock_rf, mock_train_test_split, mock_read_csv, mock_env_get):
        # Call your function
        model, rmse_rounded, mae_rounded, r2_rounded = temp_pred_model()

        # Assertions
        self.assertIsInstance(model, RandomForestRegressor)
        self.assertEqual(rmse_rounded, 1.22)
        self.assertEqual(mae_rounded, 1.22)
        self.assertTrue(pd.isna(r2_rounded), "R2 score should be NaN")

if __name__ == '__main__':
    unittest.main()

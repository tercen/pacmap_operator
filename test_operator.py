import unittest
import numpy as np
import pandas as pd
import polars as pl
from operator_funcs import reshape_data, apply_pacmap, format_results

class TestPaCMAPOperator(unittest.TestCase):
    
    def setUp(self):
        # Create sample data
        self.main_data = pl.DataFrame({
            '.y': [1.3, 0.1, 0.5, 0.8],
            '.ci': [0, 1, 0, 1],
            '.ri': [0, 0, 1, 1]
        })
        
        self.col_data = pl.DataFrame({
            'eventId': [1, 2],
            '.ci': [0, 1]
        })
        
        self.row_data = pl.DataFrame({
            'pixel_id': [1, 2],
            '.ri': [0, 1]
        })
    
    def test_reshape_data(self):
        matrix = reshape_data(self.main_data, self.col_data, self.row_data)
        self.assertEqual(matrix.shape, (2, 2))  # 2 events, 2 pixels
        
    def test_apply_pacmap(self):
        # Create a simple matrix for testing
        test_matrix = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ])
        
        result = apply_pacmap(test_matrix, n_components=2)
        self.assertEqual(result.shape, (4, 2))  # 4 samples, 2 components
        
    def test_format_results(self):
        # Create a simple embedding for testing
        embedding = np.array([
            [0.1, 0.2],
            [0.3, 0.4]
        ])
        
        result = format_results(embedding, self.col_data)
        self.assertEqual(result.shape[0], 2)  # 2 events
        self.assertEqual(result.shape[1], 5)  # PaCMAP_1, PaCMAP_2, eventId, .ci, .ri
        self.assertTrue('PaCMAP_1' in result.columns)
        self.assertTrue('PaCMAP_2' in result.columns)
        
if __name__ == '__main__':
    unittest.main()
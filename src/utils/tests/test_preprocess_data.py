import unittest
import pandas as pd
import numpy as np

from preprocess_data import preprocess_data

class TestPreprocessData(unittest.TestCase):
    def test_preprocess_data(self):
        nan = float('nan')
        
        # Testcase
        data = [
            [4.0, 2.0, 1.0, 10.0],
            [nan, nan, nan, nan],
            [4.0, 0.0, 3.0, 0.0],
            [5.0, 2.0, 1.0, 8.0],
            [5.0, 2.0, 2.0, 0.0]
        ]
        data_df = pd.DataFrame(data, columns=['q1', 'q2', 'q3', 'q4'])
        
        max_scores_data = {'q1':5.0, 'q2':2.0, 'q3':3.0, 'q4':3.0}
        max_scores_df = pd.Series(data=max_scores_data, index=['q1', 'q2', 'q3', 'q4'])

        meta_data = np.array([
            [5.0, 2.0, 3.0, 3.0],
            ['Scattergraphs and Reasoning', 'Product of Prime Factors', 'Decimal Multiplication', 'Double Brackets Problem Solving'],
            [1, 1, 1, 3],
            [171, 52, 200, 183]
        ])
        meta_df = pd.DataFrame(data=meta_data, columns=['q1', 'q2', 'q3', 'q4'])
        print(meta_df)
        
        processed_df, processed_meta_df = preprocess_data(data_df, meta_df)

        # True
        true_data = [
            [1.0, 1.0, 0.0, 1.0],
            [nan, nan, nan, nan],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 0.0]
        ]
        true_df = pd.DataFrame(true_data, columns=['q1', 'q2', 'q3', 'q4'])

        self.assertTrue(bin_df.equals(true_df))


if __name__ == '__main__':
    unittest.main()
import unittest
import os
from mock import patch
import pandas as pd

from codetemplate.src.data_processing import load_data, preprocess_dataset


class DatasetTestCase(unittest.TestCase):
    def setUp(self) -> None:
        filename = \
            '../../../../data/testset_C.csv'
        self.dataframe = load_data(filename, delimiter=';')
        self.target_var = 'productgroup'

    @patch('os.path.isfile')
    @patch('pandas.read_csv')
    def test_load_data_calls_read_csv_if_exists(self, mock_isfile, mock_read_csv):
        """
        Test for checking the invocation of the load method which calls read_csv().
        """
        # arrange
        # always return true for isfile
        os.path.isfile.return_value = True
        filename = \
            '/data/testset_C.csv'

        # act
        _ = load_data(filename, delimiter=';')
        # =================================
        # TEST SUITE
        # =================================
        # check that read_csv is called with the correct parameters
        pd.read_csv.assert_called_once_with(filename, delimiter=';')

    def test_if_nan_values_exists(self):
        """
        Check if NAN values are present in the dataset.
        """
        # =================================
        # TEST SUITE
        # =================================
        df = self.dataframe
        assert (df.isna().sum().sum()) > 0

    def test_if_null_values_exists(self):
        """
        Check whether null values are present in the dataset.
        """
        # =================================
        # TEST SUITE
        # =================================
        df = self.dataframe
        assert df.isnull().sum().sum() > 0

    def test_if_duplicates_exists(self):
        """
        Check whether duplicates present in the dataset.
        """
        # =================================
        # TEST SUITE
        # =================================
        assert self.dataframe['add_text'].nunique() < self.dataframe.shape[0]
        assert self.dataframe['main_text'].nunique() < self.dataframe.shape[0]
        assert self.dataframe['manufacturer'].nunique() < self.dataframe.shape[0]

    def test_if_data_exits_for_all_id(self):
        """
        Check if all data fields are present or not in the dataset.
        """
        # =================================
        # TEST SUITE
        # =================================
        assert self.dataframe['id'].unique().shape[0] == len(self.dataframe['id'])

    def test_if_target_has_non_binary_data(self):
        """
        Check whether the target classification is non binary.
        """
        # =================================
        # TEST SUITE
        # =================================
        assert self.dataframe[self.target_var].nunique() > 2

    def test_if_train_and_test_matrix_have_same_dimension(self):
        """
        Check whether the existing training and testing matrices
         have the same dimension or not.
        """
        # =================================
        # TEST SUITE
        # =================================
        X_train, _, X_test, _ = preprocess_dataset(self.dataframe)
        assert X_train.shape[1] == X_test.shape[1]


if __name__ == '__main__':
    unittest.main()

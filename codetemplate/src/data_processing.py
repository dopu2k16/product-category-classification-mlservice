import os
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def load_data(filename, delimiter):
    """
    Loading the dataset if the file exists.
    """
    if os.path.isfile(filename):
        df = pd.read_csv(filename, delimiter=delimiter)
        return df
    else:
        print("File not found")
        return None


def preprocess_dataset(input_data):
    """ The preprocessing selects the relevant data.

    :param input_data: Input data
    :return X: Feature selection for input data and returning the training data and test data.
    :rtype: .... """
    feature_cols = ['merged_text']
    input_data = input_data[['productgroup', 'main_text', 'add_text']].dropna()
    input_data['merged_text'] = input_data[['main_text', 'add_text']].apply(lambda x: ' '.join(x), axis=1)
    print("Columns in the data frame", input_data.columns)
    print("Input dataframe shape", input_data.shape)

    X = input_data[feature_cols]
    # Putting response variable to y
    y = input_data['productgroup']
    # Splitting the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,
                                                        test_size=0.2, random_state=100)

    return X_train, y_train, X_test, y_test


def transform_data():
    """
    Transforming into TF-IDF vectorization of the input data
    """
    vect = TfidfVectorizer()
    transformer = make_column_transformer((vect, 'merged_text'))
    return vect, transformer


def main():
    input_data = load_data('../../data/testset_C.csv')
    X_train, y_train, X_test, y_test = preprocess_dataset(input_data)
    X_train.to_csv('X_train.csv')
    X_test.to_csv('y_train.csv')
    X_test.to_csv('X_test.csv')
    y_test.to_csv('y_test.csv')


if __name__ == "__main__":
    main()

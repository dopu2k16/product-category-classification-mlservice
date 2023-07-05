from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from codetemplate.src.batch_score import batch_prediction
from codetemplate.src.data_processing import preprocess_dataset, load_data, transform_data

import warnings
warnings.filterwarnings('ignore')


def get_ml_models():
    """
    ML models to be trained.
    """
    models = dict()
    vect, transformer = transform_data()
    # Logistic regression
    model = LogisticRegression()
    models['LR'] = Pipeline(steps=[('feature_transform', transformer), ('m', model)])

    # Multinomial Naive Bayes
    model = MultinomialNB(alpha=.01)
    models['MultiNaiveBayes'] = Pipeline(steps=[('feature_transform', transformer), ('m', model)])

    # Perceptron
    model = Perceptron()
    models['Perceptron'] = Pipeline(steps=[('feature_transform', transformer), ('m', model)])

    # Decision Tree
    model = DecisionTreeClassifier()
    models['CART'] = Pipeline(steps=[('feature_transform', transformer), ('m', model)])

    # Random Forest
    model = RandomForestClassifier()
    models['RandomForest'] = Pipeline(steps=[('feature_transform', transformer), ('m', model)])

    # Multilayer Perceptron
    model = MLPClassifier(random_state=1, early_stopping=True)
    models['MLP'] = Pipeline(steps=[('feature_transform', transformer), ('m', model)])

    return models


def main():
    input_data = load_data('../../data/testset_C.csv', delimiter=';')
    X_train, y_train, X_test, y_test = preprocess_dataset(input_data)
    # getting all the implemented ml models
    models = get_ml_models()
    # getting the predictions for both the training and testing datasets
    batch_prediction(X_train, y_train, X_test, y_test, models)


if __name__ == "__main__":
    main()

from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate


def evaluate_model_predict(model, X, y, X_test, y_test):
    """
    Performs the cross validation of the ML algorithms and returns the training,
     validation scores, and test predictions.
    """
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    model.fit(X, y)
    # cross validation
    scores = cross_validate(model, X, y, scoring=('accuracy', 'f1_weighted'),
                            cv=cv, n_jobs=-1, return_train_score=True)
    # test prediction from the trained ml model
    y_test_pred = model.predict(X_test)

    # test metrics like accuracy, precision, recall, f1-score
    test_acc = metrics.accuracy_score(y_test_pred, y_test)
    test_p, test_r, test_f1, _ = precision_recall_fscore_support(y_test, y_test_pred, average='weighted')

    return scores, y_test_pred, test_acc, test_p, test_r, test_f1

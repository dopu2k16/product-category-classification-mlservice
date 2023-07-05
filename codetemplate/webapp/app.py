from flask import Flask, render_template, request, jsonify, make_response
import flask
import numpy as np
import pickle
import pandas as pd

# App definition
app = Flask(__name__, template_folder='templates')
NOT_FOUND = 'Not Found'
BAD_REQUEST = 'Bad Request'
BAD_GATEWAY = 'Bad Gateway'

# loading machine learning model
with open('../codetemplate/webapp/model/MLP.pkl', 'rb') as f:
    classifier = pickle.load(f)

'''
with open('webapp/model/model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)
'''


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': NOT_FOUND}), 404)


@app.errorhandler(400)
def bad_request(error):
    return make_response(jsonify({'error': BAD_REQUEST}), 400)


@app.errorhandler(500)
def internal_server_error(error):
    return make_response(jsonify({'error': 'Internal Server Error'}), 500)


@app.errorhandler(502)
def bad_gateway(error):
    return make_response(jsonify({'error': BAD_GATEWAY}), 502)


@app.route("/")
def index():
    return render_template('index.html', pred=0, pred_prob=0)


@app.route('/predict', methods=['POST'])
def predict():
    query = request.form['text']
    print(query)
    query = pd.DataFrame({'merged_text': query}, index=[0])
    # query = query.reindex(columns=['merged_text'])

    predictions = classifier.predict(query)
    print('User Input Predictions: {}'.format(predictions))
    pred_prob = classifier.predict_proba(query)
    print("Prediction Probability", pred_prob)

    return render_template('index.html', pred=predictions, pred_prob=np.max(pred_prob))


if __name__ == "__main__":
    app.run()

### Product Category ML Classification Web App Service (ML-as-service)

The developed model has achieved the highest test accuracy of 99.812 % and 0.998 test F1-score.

The notebook [codetemplate/notebooks/EDA.ipynb](codetemplate/notebooks/EDA.ipynb) contains the data exploration and analysis. The modeling, experiments, training,
 and evaluation of the algorithms are explained in the [codetemplate/notebooks/Modeling-Training-Evaluation.ipynb](codetemplate/notebooks/Modeling-Training-Evaluation.ipynb)  notebook.

The webapp is built on Flask framework which takes user input on the go and returns a predicted category and predicted probability.
 
`codetemplate` directory contains the source files, the unittests, the flask webapp, ipynb notebooks & web html, results for the project.

| Model                   | Training Accuracy | Validation Accuracy | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Training F1-score | Validation F1-score |
|-------------------------|-------------------|---------------------|---------------|----------------|-------------|---------------|-------------------|---------------------|
| Logistic Regression     | 99.969            | 99.755              | 99.625        | 0.99629        | 0.996       | 0.996         | 1.0               | 0.998               |
| Multinomial Naive Bayes | 100               | 99.802              | 99.688        | 0.996877       | 0.997       | 0.997         | 1.0               | 0.998               |
| Perceptron              | 100               | 99.656              | 99.688        | 0.996895       | 0.997       | 0.997         | 1.0               | 0.997               |              |
| CART                    | 100.000           | 98.296              | 99.125        | 0.991449       | 0.991       | 0.991         | 1.000             | 0.983               |
| Random Forest           | 100.000           | 99.380              | 99.500        | 0.995099       | 0.995       | 0.995         | 1.000             | 0.994               |
| MLP                     | 99.969            | 99.812              | 99.812        | 0.998127       | 0.998       | 0.998         | 1.0               | 0.998               |

### Instructions:
1. Create a virtual environment such as conda or virtualenv
```commandline
conda create --name productClass python=3.6
```
2. Install python dependencies using pip from the requirements.txt file
```commandline
conda activate productClass
pip install -r requirements.txt
```
3. Run the unittests in the `operations/tests/` in IDE to check if everything is alright. Make sure to define the exact project
root path in the `webapp/app.py` for `operation/tests/unit/test_api.py'.
4. Run the `train.py` file to run the experiments and save trained models and the result files.
5. Start the flask web app by running the `webapp/app.py`.

# Nama : Sukma Ramadhan Asri
# Kelas : Eunoia
# Orbit

import joblib
import numpy as np
from flask import Flask, render_template, request


app = Flask(__name__)


@app.route('/check-results/', methods=['GET', 'POST'])
def results():

    if request.method == "POST":

        Gender = request.form.get('Gender')
        Height = request.form.get('Height')

        try:
            prediction = preprocessDataAndPredict(Gender, Height)

            return render_template('results.html', prediction=prediction)

        except ValueError:
            return "Please Enter valid values"

    else:
        pass


def preprocessDataAndPredict(Pregnancies,  Glucose):

    # keep all inputs in array
    test_data = [Pregnancies,  Glucose]
    print(test_data)

    # convert value data into numpy array
    test_data = np.array(test_data)

    # reshape array
    test_data = test_data.reshape(1, -1)
    print(test_data)

    # open file
    file = open("./pickle/regressionlinear_weight.pkl", "rb")

    # load trained model
    trained_model = joblib.load(file)

    # predict
    prediction = np.round(trained_model.predict(test_data), 2)

    return prediction

    pass


@app.route('/')
def home():
    return render_template('Check.html')


if __name__ == '__main__':
    app.run(debug=True)

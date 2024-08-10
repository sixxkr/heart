import numpy as np
from flask import Flask, request, render_template,jsonify
import joblib

# Create flask app
flask_app = Flask(__name__)
model = joblib.load(open("heart_model_predict_rf.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict/api", methods = ["POST"])
def predict_api():
    data1 = request.form['age']
    data2 = request.form['sex']
    data3 = request.form['cp']
    data4 = request.form['trtbps']
    data5 = request.form['chol']
    data6 = request.form['fbs']
    data7 = request.form['restecg']
    data8 = request.form['thalachh']
    data9 = request.form['exng']
    data10 = request.form['oldpeak']
    data11 = request.form['slp']
    data12 = request.form['caa']
    data13= request.form['thall']
    arr = np.array([[data1, data2, data3, data4,data5,data6,data7,data8,data9,data10,data11,data12,data13]])

    # float_features = [float(x) for x in request.form.values()]
    # arr = [np.array(float_features)]


    pred = model.predict(arr)

    return jsonify({'prediction': pred.tolist()})
    
    
@flask_app.route("/predict", methods = ["POST"])
def predict():
    pred = predict_api()
    pred = pred.response

    return render_template("index.html", prediction_text = "u have heart dieses {}".format(pred[0].decode('utf8')))

if __name__ == "__main__":
    flask_app.run(debug=True)
    
    
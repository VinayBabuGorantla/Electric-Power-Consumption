import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, request, render_template
import numpy as np

from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template("home.html")
    else:
        try:
            input_series = request.form.get("recent_values")
            values = list(map(float, input_series.strip().split(',')))

            pipeline = PredictPipeline()
            prediction = pipeline.predict_next_day(np.array(values))

            return render_template("home.html", prediction=round(prediction, 4), recent_input=values)
        except Exception as e:
            return render_template("home.html", error=str(e))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

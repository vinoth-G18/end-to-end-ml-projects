from flask import Flask, request, render_template
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Create dataframe with EXACT column names used in training
            pred_df = pd.DataFrame({
                "gender": [request.form.get("gender")],
                "race/ethnicity": [request.form.get("race/ethnicity")],
                "parental level of education": [request.form.get("parental level of education")],
                "lunch": [request.form.get("lunch")],
                "test preparation course": [request.form.get("test preparation course")],
                "reading score": [float(request.form.get("reading score"))],
                "writing score": [float(request.form.get("writing score"))]
            })

            print(pred_df)  # Debug

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            return render_template('home.html', results=results[0])

        except Exception as e:
            return str(e)


if __name__ == "__main__":
    app.run(debug=True)
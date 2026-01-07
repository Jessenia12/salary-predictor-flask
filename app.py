from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ======================
# CARGAR MODELO ENTRENADO
# ======================
model = joblib.load("model/salary_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        # ======================
        # 1. RECIBIR DATOS
        # ======================
        data = {
            "Age": int(request.form["age"]),
            "Gender": request.form["gender"],
            "Education Level": request.form["education"],
            "Job Title": request.form["job"],
            "Years of Experience": float(request.form["experience"])
        }

        # ======================
        # 2. CONVERTIR A DATAFRAME
        # ======================
        input_df = pd.DataFrame([data])

        # ======================
        # 3. PREDICCIÓN REAL
        # ======================
        salary_pred = model.predict(input_df)[0]

        return render_template(
            "result.html",
            salary=int(salary_pred),
            data=data
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import numpy as np
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # ======================
        # 1. RECIBIR DATOS
        # ======================
        age = int(request.form["age"])
        gender = request.form["gender"]
        education = request.form["education"]
        job = request.form["job"]
        experience = float(request.form["experience"])

        # ======================
        # 2. CÁLCULO DEL SALARIO (EJEMPLO)
        # ======================
        salary = int(experience * 500 + 300)

        # ======================
        # 3. DATOS PARA REGRESIÓN
        # ======================
        X = np.array([1, 2, 3, 4, 5, 6])
        y = np.array([800, 1300, 1800, 2300, 2800, 3300])

        m, b = np.polyfit(X, y, 1)
        y_pred = m * X + b

        # ======================
        # 4. CREAR GRÁFICO
        # ======================
        plt.figure(figsize=(6, 4))
        plt.scatter(X, y)
        plt.plot(X, y_pred)
        plt.scatter(experience, salary)
        plt.xlabel("Experiencia")
        plt.ylabel("Salario")
        plt.title("Regresión Lineal")

        os.makedirs("static", exist_ok=True)
        plt.savefig("static/regresion.png")
        plt.close()

        # ======================
        # 5. ENVIAR TODOS LOS DATOS
        # ======================
        data = {
            "Edad": age,
            "Género": gender,
            "Nivel educativo": education,
            "Cargo": job,
            "Años de experiencia": experience
        }

        return render_template(
            "result.html",
            salary=salary,
            data=data,
            graph="regresion.png"
        )

    return render_template("index.html")


# ======================
# EJECUCIÓN DE FLASK
# ======================
if __name__ == "__main__":
    app.run(debug=True)

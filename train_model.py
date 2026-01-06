# ===============================
# 1. IMPORTAR LIBRERÍAS
# ===============================
print("Paso 1: Importando librerías...")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

print("Librerías importadas correctamente.\n")

# ===============================
# 2. CARGAR LOS DATOS
# ===============================
print("Paso 2: Cargando dataset...")

df = pd.read_csv("salarydataset.csv")

print("Dataset cargado.")
print("Primeras filas:")
print(df.head())
print("Dimensiones originales:", df.shape)
print("\nValores nulos por columna:")
print(df.isnull().sum())
print()

# ===============================
# 3. SELECCIONAR COLUMNAS RELEVANTES
# ===============================
print("Paso 3: Seleccionando columnas relevantes...")

columns_to_keep = [
    "Age",
    "Gender",
    "Education Level",
    "Job Title",
    "Years of Experience",
    "Salary"
]

df = df[columns_to_keep]

print("Columnas seleccionadas:")
print(df.columns)
print()

# ===============================
# 4. LIMPIEZA DE VALORES NULOS
# ===============================
print("Paso 4: Limpiando valores nulos...")

df = df.dropna(subset=["Salary"])

print("Dimensiones después de limpieza:", df.shape)
print()

# ===============================
# 5. DEFINIR X e y
# ===============================
print("Paso 5: Definiendo variables X e y...")

X = df.drop(columns=["Salary"])
y = df["Salary"]

print("Dimensión de X:", X.shape)
print("Dimensión de y:", y.shape)
print()

# ===============================
# 6. DEFINIR COLUMNAS NUMÉRICAS Y CATEGÓRICAS
# ===============================
print("Paso 6: Definiendo tipos de columnas...")

num_features = ["Age", "Years of Experience"]
cat_features = ["Gender", "Education Level", "Job Title"]

print("Columnas numéricas:", num_features)
print("Columnas categóricas:", cat_features)
print()

# ===============================
# 7. CREAR PIPELINES DE PREPROCESAMIENTO
# ===============================
print("Paso 7: Creando pipelines de preprocesamiento...")

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

print("Pipelines creados correctamente.\n")

# ===============================
# 8. PIPELINE COMPLETO DEL MODELO
# ===============================
print("Paso 8: Construyendo pipeline completo del modelo...")

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

print("Pipeline del modelo listo.\n")

# ===============================
# 9. DIVISIÓN TRAIN / TEST
# ===============================
print("Paso 9: Dividiendo datos en entrenamiento y prueba (80/20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Entrenamiento:", X_train.shape)
print("Prueba:", X_test.shape)
print()

# ===============================
# 10. ENTRENAMIENTO DEL MODELO
# ===============================
print("Paso 10: Entrenando el modelo...")

model.fit(X_train, y_train)

print("Modelo entrenado correctamente.\n")

# ===============================
# 11. EVALUACIÓN DEL MODELO
# ===============================
print("Paso 11: Evaluando el modelo...")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Resultados de evaluación:")
print("MAE:", mae)
print("R²:", r2)
print()

# ===============================
# 12. GUARDAR EL MODELO
# ===============================
print("Paso 12: Guardando el modelo...")

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/salary_model.pkl")

print("Modelo guardado exitosamente en model/salary_model.pkl")

# scripts/train_model.py

import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import joblib

print("--- Iniciando el script de entrenamiento ---")

df = pd.read_csv("data/calidadAireGuayaquil.csv")

df["datetime"] = pd.to_datetime(df["datetimeLocal"])
df = df.set_index("datetime")

columnas_a_eliminar = [
    "location_id", "location_name", "unit", "datetimeUtc",
    "datetimeLocal", "timezone", "latitude", "longitude",
    "country_iso", "isMobile", "isMonitor", "owner_name", "provider"
]
df_limpio = df.drop(columns=columnas_a_eliminar)
df_pivot = df_limpio.pivot_table(index="datetime", columns="parameter", values="value")

df_final = df_pivot.copy()
df_final["hora"] = df_final.index.hour
df_final["dia_semana"] = df_final.index.dayofweek
df_final["mes"] = df_final.index.month
df_final["dia_del_año"] = df_final.index.dayofyear

df_final = df_final.dropna(subset=["pm25"]) 

y = df_final["pm25"]
X = df_final.drop(columns=["pm1", "pm10", "pm25", "um003"])

split_point = int(len(X) * 0.8)
X_train, X_test = X[:split_point], X[split_point:]
y_train, y_test = y[:split_point], y[split_point:]

print(f"Entrenando modelo con {len(X_train)} registros...")

model = lgb.LGBMRegressor(random_state=42)
model.fit(X_train, y_train)

print("¡Modelo entrenado exitosamente!")

predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Evaluación del modelo - Error Cuadrático Medio (MSE): {mse:.2f}")

joblib.dump(model, "scripts/modelo_calidad_aire.pkl")
print("Modelo guardado como 'scripts/modelo_calidad_aire.pkl'")
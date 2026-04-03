
# from flask import Flask, jsonify, request
# import os
# import pickle
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
# import numpy as np

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# os.chdir(BASE_DIR)

# app = Flask(__name__)

# # Carga el modelo
# with open(os.path.join(BASE_DIR, 'ad_model.pkl'), 'rb') as f:
#     model = pickle.load(f)
# # Enruta la landing page (endpoint /)
# @app.route("/", methods=["GET"])
# def hello(): #ligado al endpoint "/" o sea
#         return "Binevenido a mi API del modelo de ventas de publicidad"

# # Enruta la funcion al endpoint /api/v1/predict
# @app.route("/api/v1/predict", methods=["GET"])
# def predict(): # Ligado al endpoint '/api/v1/predict', con el método GET
#     tv = request.args.get('tv', np.nan, type=float)
#     radio = request.args.get('radio', np.nan, type=float)
#     newspaper = request.args.get('newspaper', np.nan, type=float)

#     missing = [name for name, val in [('tv', tv), ('radio', radio), ('newspaper', newspaper)] if np.isnan(val)]

#     input_data = pd.DataFrame({'tv': [tv], 'radio': [radio], 'newspaper': [newspaper]})
#     prediction = model.predict(input_data)

#     response = {'predictions': prediction[0]}
#     if missing:
#         response['warning'] = f"Missing values imputed for: {', '.join(missing)}"

#     return jsonify(response)

# # Enruta la funcion al endpoint /api/v1/retrain
# @app.route("/api/v1/retrain", methods=["GET"])
# def retrain(): # Ligado al endpoint '/api/v1/retrain/', método GET
#     global model
#     if os.path.exists("data/Advertising_new.csv"):
#         data = pd.read_csv('data/Advertising_new.csv')
#         data.columns = [col.lower() for col in data.columns]

#         X = data.drop(columns=['sales'])
#         y = data['sales']

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#         model.fit(X_train, y_train)
#         rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
#         mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
#         model.fit(X, y)

#         with open('ad_model.pkl', 'wb') as f:
#             pickle.dump(model, f)

#         return f"Model retrained. New evaluation metric RMSE: {str(rmse)}, MAPE: {str(mape)}"
#     else:
#         return "<h2>New data for retrain NOT FOUND. Nothing done!</h2>"




# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

# 📁 Ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# 🔥 Carga del modelo
model_path = os.path.join(BASE_DIR, 'ad_model.pkl')

if not os.path.exists(model_path):
    raise FileNotFoundError("No se encontró el archivo ad_model.pkl")

with open(model_path, 'rb') as f:
    model = pickle.load(f)


# 🏠 Endpoint raíz
@app.route("/", methods=["GET"])
def hello():
    return "Bienvenido a mi API del modelo de ventas de publicidad"


# 🔮 Endpoint de predicción
@app.route("/api/v1/predict", methods=["GET"])
def predict():
    tv = request.args.get('tv', type=float)
    radio = request.args.get('radio', type=float)
    newspaper = request.args.get('newspaper', type=float)

    # 🔴 Validación de parámetros
    if tv is None or radio is None or newspaper is None:
        return jsonify({
            "error": "Faltan parámetros. Usa: tv, radio, newspaper"
        }), 400

    input_data = pd.DataFrame({
        'tv': [tv],
        'radio': [radio],
        'newspaper': [newspaper]
    })

    try:
        prediction = model.predict(input_data)[0]
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "prediction": float(prediction)
    })


# 🔄 Endpoint de reentrenamiento
@app.route("/api/v1/retrain", methods=["GET"])
def retrain():
    global model

    data_path = os.path.join(BASE_DIR, "data", "Advertising_new.csv")

    if not os.path.exists(data_path):
        return "<h2>New data for retrain NOT FOUND. Nothing done!</h2>"

    try:
        data = pd.read_csv(data_path)
        data.columns = [col.lower() for col in data.columns]

        X = data.drop(columns=['sales'])
        y = data['sales']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        model.fit(X_train, y_train)

        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        mape = mean_absolute_percentage_error(y_test, model.predict(X_test))

        # Reentrenar con todo el dataset
        model.fit(X, y)

        # Guardar modelo actualizado
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        return jsonify({
            "message": "Model retrained successfully",
            "RMSE": float(rmse),
            "MAPE": float(mape)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# 🚀 Ejecutar en local
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

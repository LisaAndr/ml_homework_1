from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import numpy as np

app = FastAPI()


# Загрузка модели скалера и декодера
with open('model.pkl', 'rb') as f:
    model = joblib.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = joblib.load(f)

with open('encoder.pkl', 'rb') as f:
    encoder = joblib.load(f)

class Item(BaseModel):
    name: str
    year: int
    # selling_price: int таргет подаваться не должен
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

def import_data(item: Item) -> pd.DataFrame:
    # Преобразование входящих данных в DataFrame
    data = {
        "name": item.name,
        "year": item.year,
        "km_driven": item.km_driven,
        "fuel": item.fuel,
        "seller_type": item.seller_type,
        "transmission": item.transmission,
        "owner": item.owner,
        "mileage": item.mileage,
        "engine": item.engine,
        "max_power": item.max_power,
        "torque": item.torque,
        "seats": item.seats
    }
    return pd.DataFrame([data])

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    data[['mileage_value', 'mileage_unit']] = data['mileage'].str.extract(r'([\d.]+)\s*([a-zA-Z/ ]+)')
    data[['engine_value', 'engine_unit']] = data['engine'].str.extract(r'([\d.]+)\s*([a-zA-Z/ ]+)')
    data[['max_power_value', 'max_power_unit']] = data['max_power'].str.extract(r'([\d.]+)\s*([a-zA-Z/ ]+)')
    data['mileage_value'] = data['mileage_value'].astype(float)
    data['engine_value'] = data['engine_value'].astype(float)
    data['max_power_value'] = data['max_power_value'].astype(float)
    data['car_brand'] = data['name'].str.split(' ').str[0]
    data = data.drop(['mileage', 'mileage_unit', 'engine', 'engine_unit',
                     'max_power', 'max_power_unit', 'name', "torque"], axis=1)
    data = data.rename(columns={'mileage_value': 'mileage',
                                'engine_value': 'engine',
                                'max_power_value': 'max_power'})
    return data


def encoding(data: pd.DataFrame) -> np.ndarray:
    numeric_features = ['year', 'km_driven', 'mileage', 'engine', 'max_power']
    data_numeric = data[numeric_features]
    categorical_features = ['fuel', 'seller_type',  'transmission', 'owner', 'car_brand', 'seats']
    data_categorical = data[categorical_features]
    data_numeric_scaled = scaler.transform(data_numeric)
    data_cat_scaled = pd.concat([pd.DataFrame(data_numeric_scaled, columns=numeric_features),
                            data_categorical.reset_index(drop=True)], axis=1)
    data_encoded = encoder.transform(data_cat_scaled[categorical_features])
    data_encoded_df = pd.DataFrame(data_encoded,
                                 columns=encoder.get_feature_names_out(categorical_features))
    return data_cat_scaled.drop(columns=categorical_features).reset_index(drop=True).join(data_encoded_df)
    


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    try:
        # Обработка данных
        input_data = import_data(item)
        data = preprocess_data(input_data)
        X = encoding(data)
        # Предсказание
        prediction = model.predict(X)
        return float(prediction[0])  # Возврат результата предсказания
    except Exception as e:
        raise HTTPException(status_code=500, detail=e)

@app.post("/predict_items")
def predict_items(items: Items) -> List[float]:
    input_data = []
    try:
        for item in items.objects:
            input_data.append(import_data(item))
        all_data = pd.concat(input_data, axis=0)
        data = preprocess_data(all_data)
        X = encoding(data)
        prediction = model.predict(X)
        return prediction.tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=e)

if __name__ == "__main__":
   uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
from pandas import DataFrame
from joblib import load
from pydantic import BaseModel, ValidationError
from fastapi import FastAPI, HTTPException

model = load("artifacts/model.pkl")

app = FastAPI()

COLUMNS = [
    'Age', 'Gender', 'Ethnicity', 'ParentalEducation',
    'StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport',
    'Extracurricular', 'Sports',
]


class DataPredict(BaseModel):
    data_to_predict: list[list] = [
        [17, 1, 0, 2, 19.83, 7, 1, 2, 0, 0],
        [15, 0, 2, 3, 4.21, 26, 0, 2, 0, 0],
    ]


@app.post("/predict")
def predict(request: DataPredict):
    try:
        df_data = DataFrame(request.data_to_predict, columns=COLUMNS)
        prediction = model.predict(df_data)
        return {"prediction": prediction.tolist()}
    except ValidationError as ve:
        raise HTTPException(status_code=400, detail=ve.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def home():
    return {'Taller 3': 'MLOps'}

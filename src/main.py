import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel
from pydantic.types import confloat, conint

from data_controller import DataController
from model_controller import ModelController


class Passenger(BaseModel):
    passenger_class: conint(ge=1, le=3)
    sex: bool
    age: confloat(ge=0.0)
    siblings_and_spouses: conint(ge=0)
    parents_and_children: conint(ge=0)


app = FastAPI()


@app.get("/")
def index():
    return {"test": "test"}


@app.post("/predict")
def predict(passenger: Passenger):
    mc = ModelController()
    mc.load_model()

    x = np.array([[
        passenger.passenger_class,
        passenger.sex,
        passenger.age,
        passenger.siblings_and_spouses,
        passenger.parents_and_children
    ]])
    pred = mc.predict(x)
    print(pred)
    return {"prediction": str(pred)}


if __name__ == "__main__":
    data_controller = DataController()
    model_controller = ModelController()

    X_train, X_test, y_train, y_test = data_controller.get_datasets()
    model_controller.fit(X_train, y_train)
    model_controller.evaluate(X_test, y_test)
    model_controller.save_model()

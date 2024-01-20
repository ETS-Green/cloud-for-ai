from data_controller import DataController
from model_controller import ModelController

if __name__ == "main":
    data_controller = DataController()
    model_controller = ModelController()

    X_train, X_test, y_train, y_test = data_controller.get_datasets()
    model_controller.fit(X_train, y_train)
    model_controller.evaluate(X_test, y_test)

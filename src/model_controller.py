import matplotlib.pyplot as plt

from joblib import dump, load
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error, r2_score


class ModelController:
    def __init__(self):
        self.model = linear_model.LogisticRegression()
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, x, y):
        """Trains the stored data on the given datasets"""
        self.scaler.fit(x)
        x = self.scaler.transform(x)

        self.model.fit(x, y)
        self.fitted = True

    def predict(self, x):
        """Returns a prediction run by the stored model on the given data"""
        if not self.fitted:
            raise Exception("Model not yet trained")

        x = self.scaler.transform(x)
        return self.model.predict(x)

    def evaluate(self, x_test, y_test):
        """Evaluates the model based on the given test dataset"""
        pred = self.predict(x_test)
        print("Coefficients: \n", self.model.coef_)
        print("MSE: %.2f" % mean_squared_error(y_test, pred))
        print("R2 Score: %.2f" % r2_score(y_test, pred))

        # draw the confusion matrix
        cm = confusion_matrix(y_test, pred, labels=self.model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.model.classes_)
        disp.plot()
        plt.show()

    def save_model(self):
        """Saves the trained model"""
        dump(self.model, "./model/logistic_regression.joblib")
        dump(self.scaler, "./model/scaler.joblib")
        print("Model saved successfully")

    def load_model(self):
        """Saves the trained model"""
        self.model = load("./model/logistic_regression.joblib")
        self.scaler = load("./model/scaler.joblib")
        self.fitted = True
        print("Model loaded successfully")

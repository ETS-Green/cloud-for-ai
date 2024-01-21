import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class DataController:
    def __init__(self):
        self.df = None
        self.X = None
        self.y = None

        self.scaler = StandardScaler()

        self.__load_data()
        self.__preprocess_data()
        self.__split_data()

    def get_datasets(self, test_size: float = 0.2, random_state: int = None):
        """Creates training and test datasets from the stored data"""
        return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

    def __load_data(self):
        """loads a CSV into the controller"""
        self.df = pd.read_csv("../data/Titanic-Dataset.csv")
        print("Data loaded successfully")

    def __preprocess_data(self):
        """removes null values and redundant columns from the dataset"""
        self.df = self.df.drop(columns=["PassengerId", "Name", "Ticket", "Fare", "Cabin", "Embarked"])
        self.df = self.df.dropna()

        self.df['Sex'] = self.df['Sex'].astype('category')
        cat_columns = self.df.select_dtypes('category').columns
        self.df[cat_columns] = self.df[cat_columns].apply(lambda x: x.cat.codes)

        print("Data processed successfully")

    def __split_data(self):
        """Splits the data into an array of features and an array of labels"""
        self.X = self.df.drop('Survived', axis=1)
        self.X = np.array(self.X)

        self.y = self.df['Survived']
        self.y = np.array(self.y)

        # Erase dataframe to save memory
        self.df = None

        print("Data split successfully")

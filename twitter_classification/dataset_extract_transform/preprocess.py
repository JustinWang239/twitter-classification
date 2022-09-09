import pandas as pd
import numpy as np
from typing import Tuple

class Loader():
    def __init__(self, path: str):
        self.path = path

    def load(self) -> None:
        self.df = pd.read_csv(self.path)

    def preprocess(self) -> None:
        ## Check for any null values
        self.df.isnull().values.any()

        self.df = self.df[['class', 'tweet']]

        from sklearn.model_selection import train_test_split
        X = self.df['tweet']
        y = self.df['class']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        # def check_ratio(feat_df: pd.DataFrame, df: pd.DataFrame, header: str) -> None:
        #     print(header + ' : {0} ({1:0.2f}%)'.format(len(feat_df), (len(feat_df)/len(df)) * 100.0))

        # # Verify split ratios
        # print('{0:0.2f}% in training set'.format((len(X_train)/len(self.df.index)) * 100))
        # print('{0:0.2f}% in test set'.format((len(X_test)/len(self.df.index)) * 100))
        # print('')
        # check_ratio(self.df.loc[self.df['class'] == 0], self.df.index, 'Original Hate Speech')
        # check_ratio(self.df.loc[self.df['class'] == 1], self.df.index, 'Original Offensive')
        # check_ratio(self.df.loc[self.df['class'] == 2], self.df.index, 'Original Neither')
        # print('')
        # check_ratio(y_train[y_train[:] == 0], y_train, 'Training Hate Speech')
        # check_ratio(y_train[y_train[:] == 1], y_train, 'Training Offensive')
        # check_ratio(y_train[y_train[:] == 2], y_train, 'Training Neither')
        # print('')
        # check_ratio(y_test[y_test[:] == 0], y_test, 'Test Hate Speech')
        # check_ratio(y_test[y_test[:] == 1], y_test, 'Test Offensive')
        # check_ratio(y_test[y_test[:] == 2], y_test, 'Test Neither')

        self.train_set = X_train.values
        self.test_set = X_test.values
        self.train_labels = y_train.values
        self.test_labels = y_test.values

    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.train_set, self.train_labels

    def get_testing_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.test_set, self.test_labels

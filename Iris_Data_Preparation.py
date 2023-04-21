import pandas as pd
import numpy as np
import os

class Iris_data:
    def __init__(self):
        self.X, Y = self.iris_training()
        self.Y = Y.T

    def iris_training(self):
        CURRENT_DIR = os.path.dirname(__file__)
        file_path = os.path.join(CURRENT_DIR, 'Iris.csv')
        df = pd.read_csv(file_path)

        target_col = 'Species'

        rez = np.array(df[target_col])
        y_uniques = df[target_col].unique()
        dict_rez = {elem: i+1 for i, elem in enumerate(y_uniques)}
        Y = np.array([dict_rez[elem] for elem in rez]).reshape(len(rez), 1).T
        Y = np.array(pd.get_dummies(Y[0]))

        
        X = []
        for col in df:
            if col != target_col and col !='Id':
                X += [df[col]]
        X = np.array(X)
        return X, Y
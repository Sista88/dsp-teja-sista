import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import joblib


mmscaler = joblib.load('models/mmscaler.joblib')
onehotencoder = joblib.load('models/onehotencoder.joblib')
lin_reg_model = joblib.load('models/model.joblib')

houses_prediction = pd.read_csv('train (2).csv')


def build_model(data: pd.DataFrame) -> dict[str, str]:
    X, y = data[['MSSubClass', 'MSZoning', 'Utilities',
                 'OverallQual', 'OverallCond',
                 'GarageArea', 'BldgType']], data['SalePrice']
    X['GarageArea'] = mmscaler.transform(X[['GarageArea']])
    filtered_transform_1 = pd.DataFrame(onehotencoder.transform(
        X[['MSZoning', 'BldgType', 'Utilities']]).toarray())
    filtered_transform_1.columns = onehotencoder.get_feature_names(
        ['MSZoning', 'BldgType', 'Utilities'])
    X = pd.concat([X, filtered_transform_1], axis=1)
    X = X.drop(columns=['MSZoning', 'Utilities', 'BldgType'], axis=1)
    X = X.dropna()
    new_df_2 = pd.merge(X, y, how='inner', right_index=True, left_index=True)
    y = new_df_2['SalePrice']
    X = new_df_2.drop(columns=['SalePrice'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    y_pred = abs(lin_reg_model.predict(X_test))

    def compute_rmsle(y_test: np.ndarray, y_pred: np.ndarray,
                      precision: int = 2) -> float:
        rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
        return round(rmsle, precision)

    return dict({'rmse': compute_rmsle(y_test, y_pred)})


build_model(houses_prediction)

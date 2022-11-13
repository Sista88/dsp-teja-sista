import pandas as pd
import numpy as np
import joblib


mmscaler = joblib.load('models/mmscaler.joblib')
onehotencoder = joblib.load('models/onehotencoder.joblib')
lin_reg_model = joblib.load('models/model.joblib')


test_df = pd.read_csv('test (1).csv')


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    test_data = input_data[
        ['MSSubClass', 'MSZoning', 'Utilities',
         'OverallQual', 'OverallCond',
         'GarageArea', 'BldgType']]
    test_data['GarageArea'] = mmscaler.transform(test_data[['GarageArea']])
    filtered_transform_1 = pd.DataFrame(
        onehotencoder.transform(test_data[['MSZoning', 'BldgType',
                                           'Utilities']]).toarray())
    filtered_transform_1.columns = onehotencoder.get_feature_names(
        ['MSZoning', 'BldgType', 'Utilities'])
    test_data = pd.concat([test_data, filtered_transform_1], axis=1)
    test_data = test_data.drop(columns=['MSZoning', 'Utilities',
                                        'BldgType'], axis=1)
    test_data = test_data.dropna()
    y_test_data = abs(lin_reg_model.predict(test_data))
    return y_test_data

# the model and all the data preparation objects (encoder, etc)
# should be loaded from the models folder


make_predictions(test_df)

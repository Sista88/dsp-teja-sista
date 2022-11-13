import pandas as pd
import joblib


mmscaler = joblib.load('models/mmscaler.joblib')
onehotencoder = joblib.load('models/onehotencoder.joblib')
lin_reg_model = joblib.load('models/model.joblib')

# Cleaning "train.csv" data

houses_prediction = pd.read_csv('train (2).csv')
y = houses_prediction['SalePrice']
X = houses_prediction[['MSSubClass', 'MSZoning',
                       'Utilities', 'OverallQual', 'OverallCond',
                       'GarageArea', 'BldgType']]
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


# Cleaning "test.csv" data

test_df = pd.read_csv('test (1).csv')
test_data = test_df[['MSSubClass', 'MSZoning',
                     'Utilities', 'OverallQual', 'OverallCond',
                     'GarageArea', 'BldgType']]
test_df['GarageArea'] = mmscaler.transform(test_data[['GarageArea']])
filtered_transform_1 = pd.DataFrame(onehotencoder.transform(
    test_data[['MSZoning', 'BldgType', 'Utilities']]).toarray())
filtered_transform_1.columns = onehotencoder.get_feature_names(
    ['MSZoning', 'BldgType', 'Utilities'])
test_data = pd.concat([test_data, filtered_transform_1], axis=1)
test_data = test_data.drop(columns=['MSZoning', 'Utilities',
                                    'BldgType'], axis=1)
test_data = test_data.dropna()

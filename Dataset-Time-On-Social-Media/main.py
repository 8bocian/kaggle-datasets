import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import pandas as pd
import keras.layers
import keras

df = pd.read_csv('dummy_data.csv')

# for column in df.columns:
#     plt.pie(df[column].value_counts(), labels=df[column].unique(), autopct='%.1f%%')
#     plt.title(column)
    # plt.show()


encoders = {}
for i in df.columns[:-1]:
    encoders[i] = LabelEncoder()
    encoders[i].fit(df[i])
    df[i] = encoders[i].transform(df[i])

print(df.transpose())

scalers = {}
for i in ['age', 'income']:
    scalers[i] = MinMaxScaler()
    scalers[i].fit(df[i].values.reshape((-1, 1)))
    df[i] = scalers[i].transform(df[i].values.reshape((-1, 1)))

X = df.drop(columns=['time_spent'])
y = df['time_spent']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=420)

forest_clf = RandomForestRegressor()
params = {
    "n_estimators": [200, 400, 600, 800],
    'max_depth': [50, 70, 90, 110],
    'max_features': [2, 4, 6],
    'min_samples_leaf': [3, 5, 7],
    'min_samples_split': [10, 12, 14]
}
grid_search = GridSearchCV(estimator=forest_clf, param_grid=params, cv=3, verbose=2, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)
print(grid_search.best_params_)
print(mean_squared_error(y_true=y_test, y_pred=y_pred))
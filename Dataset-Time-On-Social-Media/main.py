import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import pandas as pd
import keras.layers
import keras.losses
import keras.optimizers
import keras.metrics
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
df['Owns_Car'] = df['Owns_Car'].apply(lambda x: 1 if x == "True" else 0)

scalers = {}
for i in ['age', 'income']:
    scalers[i] = MinMaxScaler()
    scalers[i].fit(df[i].values.reshape((-1, 1)))
    df[i] = scalers[i].transform(df[i].values.reshape((-1, 1)))

print(df.transpose())
df.astype(np.float32)
X = df.drop(columns=['time_spent'])
y = df['time_spent']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=420)

input_shape = len(X.columns)

inputs = keras.Input(shape=(input_shape,))
x = keras.layers.Dense(int(input_shape * 2), activation='relu')(inputs)
x = keras.layers.Dense(input_shape, activation='relu')(x)
outputs = keras.layers.Dense(1)(x)

model = keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())
model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    metrics=[keras.metrics.MeanSquaredError()],
)

model.fit(X_train, y_train, epochs=50, batch_size=16)
model.evaluate(X_test, y_test, verbose=2)



# forest_clf = RandomForestRegressor()
# params = {
#     "n_estimators": [200, 400, 600, 800],
#     'max_depth': [50, 70, 90, 110],
#     'max_features': [2, 4, 6],
#     'min_samples_leaf': [3, 5, 7],
#     'min_samples_split': [10, 12, 14]
# }
# grid_search = GridSearchCV(estimator=forest_clf, param_grid=params, cv=3, verbose=2, scoring='neg_mean_squared_error')
# grid_search.fit(X_train, y_train)
# y_pred = grid_search.predict(X_test)
#
# print(grid_search.best_params_)
# print(mean_squared_error(y_true=y_test, y_pred=y_pred))

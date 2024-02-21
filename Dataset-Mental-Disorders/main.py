import os
import matplotlib.pyplot as plt
import numpy as np
import keras.losses
import keras.layers
import keras.utils
import keras.optimizers
import keras.metrics
import pandas as pd
import seaborn
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

print(os.system('ls'))
df = pd.read_csv('Dataset-Mental-Disorders/Dataset-Mental-Disorders.csv')
df.drop(columns=['Patient Number'], inplace=True)

range_columns = [
    "Sadness",
    "Euphoric",
    "Exhausted",
    "Sleep dissorder"
]

boolean_columns = [
    "Mood Swing",
    "Suicidal thoughts",
    "Anorxia",
    "Authority Respect",
    "Try-Explanation",
    "Aggressive Response",
    "Ignore & Move-On",
    "Nervous Break-down",
    "Admit Mistakes",
    "Overthinking"
]

numeric_columns = [
    "Sexual Activity",
    "Concentration",
    "Optimisim"
]

ranges_encoding = {
    "Most-Often": 3,
    "Usually": 2,
    "Sometimes": 1,
    "Seldom": 0
}

print(df.transpose())

for numeric_column in numeric_columns:
    df[numeric_column] = df[numeric_column].apply(lambda value: int(value[0].strip()))

for boolean_column in boolean_columns:
    df[boolean_column] = df[boolean_column].apply(lambda value: True if value.lower().strip() == "yes" else False)

for range_column in range_columns:
    df[range_column] = df[range_column].apply(lambda value: ranges_encoding[value])

encodings = list(df['Expert Diagnose'].unique())
encodings.sort()
df['Expert Diagnose'] = df['Expert Diagnose'].apply(lambda x: encodings.index(x))
df = df.astype(np.int32)
X = df.drop(columns='Expert Diagnose')
y = df['Expert Diagnose']


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, shuffle=True, random_state=42)

input_shape = len(X.columns)

inputs = keras.Input(shape=(input_shape,))
x = keras.layers.Dense(int(input_shape * 2), activation='relu')(inputs)
x = keras.layers.Dense(input_shape, activation='relu')(x)
outputs = keras.layers.Dense(len(encodings), activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

model.fit(X_train, y_train, epochs=50, batch_size=16)
model.evaluate(X_test, y_test, verbose=2)
y_pred = model.predict(X_test)
y_pred = [np.argmax(row) for row in y_pred]

print(classification_report(y_test, y_pred))

cf_matrix = confusion_matrix(y_test, y_pred)
seaborn.heatmap(cf_matrix, annot=True)
plt.show()

# Dataset Link
# https://www.kaggle.com/datasets/cid007/mental-disorder-classification/data
#
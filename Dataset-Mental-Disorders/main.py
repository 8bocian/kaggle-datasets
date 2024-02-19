import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

df = pd.read_csv('Dataset-Mental-Disorders.csv')

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

X = df.drop(columns='Expert Diagnose')
y = df['Expert Diagnose']


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=42)

param_grid = {
    'bootstrap': [True, False],
    'max_depth': [50, 70, 90, 110],
    'max_features': [2, 4, 6],
    'min_samples_leaf': [3, 5, 7],
    'min_samples_split': [10, 12, 14],
    'n_estimators': [100, 200, 300, 400, 500]
}

forest_clf = RandomForestClassifier()
grid_search = GridSearchCV(estimator=forest_clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)

print(classification_report(y_test,y_pred))

# Dataset Link
# https://www.kaggle.com/datasets/cid007/mental-disorder-classification/data
#
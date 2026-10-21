from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

#data handling
test_data = pd.read_csv(r'datasets\titanic\test.csv')
train_data = pd.read_csv(r'datasets\titanic\train.csv')

train_y = train_data['Survived']
train_x = train_data.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin'])

test_data = test_data.drop(columns=['Name', 'Ticket', 'Cabin'])

#ml
preprossesor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']),

    ('labels', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ]), ['Sex', 'Embarked'])
])

pipe = Pipeline([
    ('preprossesor', preprossesor),
    ('model', LogisticRegression(max_iter=10000))
])


grid_perams = {
    'model__C': [0.01, 0.1, 1.0, 10.0, 100.0],
    'model__solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
}

grid = GridSearchCV(
    param_grid=grid_perams,
    estimator=pipe,
    cv=5,
    n_jobs=-1,
    scoring='accuracy'
)

#fitting grid
grid.fit(train_x, train_y)

#getting best model
model = grid.best_estimator_
preds = model.predict(test_data)

new_data = {
    'PassengerId': test_data['PassengerId'],
    'Survived': preds
}
df = pd.DataFrame(new_data)
#print
print(new_data)

#save to csv
df.to_csv('predictions.csv', index=False)


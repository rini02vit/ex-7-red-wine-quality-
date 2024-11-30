import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import random as rn

data = pd.read_csv('/content/winequality-red.csv', sep=',')

print(data.columns)
X = data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides']]
Y = (data['quality'] >= 5).astype(int)
rn.seed(1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=1)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)


print(f"Accuracy: {accuracy_score(Y_test, Y_pred)}")


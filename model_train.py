import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


data = pd.read_csv('Iris.csv')

data.drop('Id', axis=1, inplace=True)
X = data.drop('Species',axis=1)
y = data["Species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)

print('Jumlah Data Latih: ', len(X_train))
print('Jumlah Data Uji: ', len(X_test))

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
knn_test = knn.predict(X_test)

accuracy_score(y_test, knn_test)

print('Akurasi: ', accuracy_score(y_test, knn_test))

print(knn_test)
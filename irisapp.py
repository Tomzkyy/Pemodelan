import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()


st.title("Iris Flower Classification App using K-Nearest Neighbors (KNN) Algorithm")

img = Image.open("irisall.jpg")
st.image(img, use_column_width=False)

a = float(st.number_input("Sepal Length"))
b = float(st.number_input("Sepal Width"))
c = float(st.number_input("Petal Length"))
d = float(st.number_input("Petal Width"))

btn = st.button("Predict")

if btn:
    prediksi = np.array([a,b,c,d]).reshape(1,-1)
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(iris.data, iris.target)
    knn_test = knn.predict(prediksi)
    print(knn_test)
    st.subheader(("Iris " + iris.target_names[knn_test[0]]))

    if knn_test[0] == 0:
        st.image("irissetosa.jpg")
    elif knn_test[0] == 1:
        st.image("irisversicolor.jpg")
    else:
        st.image("irisvirginica.jpg")


    

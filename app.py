# Importar los paquetes necesarios
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objs as go

# Cargar los datos
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target'] = df['target'].apply(lambda x: iris.target_names[x])

# Entrenar el modelo
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X, y)

# Crear la aplicación Shiny
import streamlit as st
st.set_page_config(page_icon="TA_OSF.png", page_title="TA-Modelo-Iris")

st.title("Modelo de Machine Learning para Iris Dataset")

# Sidebar para selección de parámetros
st.sidebar.header("Ingrese los valores de las características de la flor")
def user_input():
    sepal_length = st.sidebar.slider('Longitud del sepalo', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Ancho del sepalo', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Longitud del pétalo', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Ancho del pétalo', 0.1, 2.5, 0.2)
    data = {'Longitud del sepalo': sepal_length,
            'Ancho del sepalo': sepal_width,
            'Longitud del pétalo': petal_length,
            'Ancho del pétalo': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features
df_input = user_input()

df_input = df_input.rename(columns={'Ancho del pétalo': 'petal width (cm)', 'Ancho del sepalo': 'sepal width (cm)', 'Longitud del pétalo': 'petal length (cm)', 'Longitud del sepalo': 'sepal length (cm)'})

# Predicción del modelo
prediction = rfc.predict(df_input)
proba = rfc.predict_proba(df_input)[0]

COLOR_RED = "#FF4B4B"
COLOR_BLUE = "#1C83E1"
COLOR_CYAN = "#00C0F2"
color_resultado = COLOR_CYAN
# Mostrar resultados
st.subheader('Predicción')
st.title(':blue[La planta es ]'+ prediction[0])

st.subheader('Distribución Probabilidades por clase')

# Histograma con las probabilidades de cada clase
classes = ['setosa', 'versicolor', 'virginica']
proba_df = pd.DataFrame({'class': classes, 'probability': proba})
fig2 = px.histogram(proba_df, x='class', y='probability', title='Probabilidad de cada clase')
st.plotly_chart(fig2)

st.subheader('Distribución de la variable objetivo')
fig = px.histogram(df, x='target', title='Distribución de la variable objetivo')
st.plotly_chart(fig)
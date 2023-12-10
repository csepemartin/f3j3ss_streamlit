
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st

goals = pd.read_csv('https://github.com/csepemartin/f3j3ss_streamlit/blob/main/Goals.csv?raw=true')
attempts = pd.read_csv('https://github.com/csepemartin/f3j3ss_streamlit/blob/main/Attempts.csv?raw=true')

goals['id'] = goals['player_name'] + goals['club']
attempts['id'] = attempts['player_name'] + attempts['club']

attempts = attempts.drop(['serial','club','position','player_name'],axis = 1)
goals = goals[['goals','id']]

X = pd.merge(goals,attempts,on='id',how='inner')

y = X['goals']

X = X.drop(['id','goals'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

st.sidebar.header('Input Parameters')

feature1 = st.sidebar.slider('Feature 1', min_value=0.0, max_value=10.0, value=5.0)
feature2 = st.sidebar.slider('Feature 2', min_value=0.0, max_value=10.0, value=5.0)

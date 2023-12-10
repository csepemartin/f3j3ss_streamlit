
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st

goals = pd.read_csv('https://github.com/csepemartin/f3j3ss_streamlit/blob/main/Goals.csv?raw=true')
attempts = pd.read_csv('https://github.com/csepemartin/f3j3ss_streamlit/blob/main/Attempts.csv?raw=true')

goals['id'] = goals['player_name'] + goals['club']
attempts['id'] = attempts['player_name'] + attempts['club']

attempts = attempts.drop(['serial','club','total_attempts','player_name'],axis = 1)
goals = goals[['goals','id']]

position_mapping = {'Defender': 1, 'Midfielder': 2, 'Forward': 3}

attempts['position'] = attempts['position'].map(position_mapping)

X = pd.merge(goals,attempts,on='id',how='inner')

y = X['goals']

X = X.drop(['id','goals'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

st.sidebar.header('Input Parameters')

position = st.sidebar.slider('Position', min_value=0.0, max_value=10.0, value=5.0)
on_target = st.sidebar.slider('On_target', min_value=0.0, max_value=30, value=10)
off_target = st.sidebar.slider('Off_target', min_value=0.0, max_value=30, value=5.0)
blocked = st.sidebar.slider('Blocked', min_value=0.0, max_value=30, value=5.0)
match_played = st.sidebar.slider('Match_played', min_value=0.0, max_value=13, value=5.0)

st.title('Linear Regression Model Deployment')

if st.sidebar.button('Predict'):
    prediction = linear_regression.predict([[position,on_target,off_target,blocked,match_played]])
    st.sidebar.text(f'Prediction: {prediction[0]}')

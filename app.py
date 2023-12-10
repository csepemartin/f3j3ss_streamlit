
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st

goals = pd.read_csv('C:\Adat\Goals.csv')
attempts = goals = pd.read_csv('C:\Adat\Attempts.csv')

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

# Example input fields for a linear regression model
# You can modify this based on your model's input features
feature1 = st.sidebar.slider('Feature 1', min_value=0.0, max_value=10.0, value=5.0)
feature2 = st.sidebar.slider('Feature 2', min_value=0.0, max_value=10.0, value=5.0)
# ...

# Create a button to trigger the prediction
if st.sidebar.button('Predict'):
    # Perform prediction using the input features
    prediction = linear_regression.predict([[feature1, feature2]])  # Modify based on your model's input
    st.sidebar.text(f'Prediction: {prediction[0]}')

st.title('Linear Regression Model Deployment')


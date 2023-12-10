
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

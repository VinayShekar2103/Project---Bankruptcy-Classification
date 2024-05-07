import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


"""***Importing and reading the file***"""

data=pd.read_excel(r"C:\Users\vinay\OneDrive\Desktop\Project\bankruptcy-prevention.xlsx")

data

"""**Here we can see that we have 250 rows and only 1 column which consists of all the values which are separated by semi colon(;). So we will treat this in the further codes.**"""

data=data.iloc[:,0].str.split(';',expand=True)
data.head()

data.columns=['industrial_risk', 'management_risk', 'financial_flexibility', 'credibility', 'competitiveness', 'operating_risk', 'class']


"""**The dtypes of all the columns are in object type, we have to convert these object dtypes to float dtype so that we can perform analysis.**"""

data['industrial_risk'] = pd.to_numeric(data['industrial_risk'], errors='coerce')
data['management_risk'] = pd.to_numeric(data['management_risk'], errors='coerce')
data['financial_flexibility'] = pd.to_numeric(data['financial_flexibility'], errors='coerce')
data['credibility'] = pd.to_numeric(data['credibility'], errors='coerce')
data['competitiveness'] = pd.to_numeric(data['competitiveness'], errors='coerce')
data['operating_risk'] = pd.to_numeric(data['operating_risk'], errors='coerce')

#Encoding

labelencoder=LabelEncoder()
data['class']=labelencoder.fit_transform(data['class'])

print(data['class'].dtypes)
data['class'].value_counts()

"""Here,

non-bankruptcy is 1

bankruptcy is 0     
"""


"""***Splitting the data for training and testing purpose.***"""

x = data.iloc[:,:-1]
y = data.iloc[:,-1]

RF_f = RandomForestClassifier(n_estimators = 100, max_features='auto')
RF_f.fit(x,y)

pred_f=RF_f.predict(x)

print(classification_report(y,pred_f))

import streamlit as st

st.title('Bankruptcy Prevention - Random forest - Group 6')
st.sidebar.header('User Input Parameters')

def user_input_features():
    industrial_risk = st.sidebar.selectbox('industrial_risk',('1','0','0.5'))
    managment_risk = st.sidebar.selectbox('management_risk',('1','0','0.5'))
    financial_flexibility = st.sidebar.selectbox('financial_flexibility',('1','0','0.5'))
    credibility = st.sidebar.selectbox('credibility',('1','0','0.5'))
    competitiveness = st.sidebar.selectbox('competitiveness',('1','0','0.5'))
    operating_risk = st.sidebar.selectbox('operating_risk',('1','0','0.5'))
    Class = st.sidebar.selectbox('class',('1','0','0.5'))
    data = {'industrial_risk':industrial_risk,
            'management_risk':managment_risk,
            'financial_flexibility':financial_flexibility,
            'credibility':credibility,
            'competitiveness':competitiveness,
             'operating_risk':operating_risk}
    features = pd.DataFrame(data,index = [0])
    return features

df = user_input_features()
st.subheader('User Input parameters')
st.write(df)


RF_f = RandomForestClassifier(n_estimators = 100, max_features='auto')
RF_f.fit(x,y)

prediction=RF_f.predict(df)
prediction_prob=RF_f.predict_proba(df)

st.subheader('Predicted Result')
st.write('Yes' if prediction_prob[0][1] > 0.7 else 'No')

st.subheader('Prediction Probability')
st.write(prediction_prob)






import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)

def preprocess_data(df):
    imputer = SimpleImputer(strategy='median')
    df['Age'] = imputer.fit_transform(df[['Age']])
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    encoder = OneHotEncoder(sparse_output=False)
    embarked_encoded = encoder.fit_transform(df[['Embarked']])
    embarked_df = pd.DataFrame(embarked_encoded, columns=encoder.get_feature_names_out(['Embarked']))
    
    df = df.drop('Embarked', axis=1)
    df = pd.concat([df, embarked_df], axis=1)
    
    scaler = StandardScaler()
    df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    df = df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
    
    return df

# Streamlit app
st.title('Titanic Feature Engineering')
st.write('This app demonstrates feature engineering techniques on the Titanic dataset.')

data_load_state = st.text('Loading data...')
data = load_data()
data_load_state.text('Loading data...done!')

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

data = preprocess_data(data)

if st.checkbox('Show processed data'):
    st.subheader('Processed data')
    st.write(data)

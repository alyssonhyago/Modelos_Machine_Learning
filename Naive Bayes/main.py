'''

Modelo de apredizagem Naive Bayes

Autor: Alysson Hyago Pereira de Oliveira

Data:13/10/2022

'''

#Base risco de crédito

import pandas as pd

base_risco_credito = pd.read_csv('./Bases de dados/risco_credito.csv' )


#Pré-processamento

X_risco_credito = base_risco_credito.iloc[:, 0:4].values
y_risco_credito = base_risco_credito.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder # transofrma os categóricos em numeros

label_encoder_historia = LabelEncoder()
label_encoder_divida = LabelEncoder()
label_encoder_garantia = LabelEncoder()
label_encoder_renda = LabelEncoder()

X_risco_credito[:,0] = label_encoder_historia.fit_transform(X_risco_credito[:,0])
X_risco_credito[:,1] = label_encoder_divida.fit_transform(X_risco_credito[:,1])
X_risco_credito[:,2] = label_encoder_garantia.fit_transform(X_risco_credito[:,2])
X_risco_credito[:,3] = label_encoder_renda.fit_transform(X_risco_credito[:,3])

import pickle
f = open('risco_credito.pkl', 'wb')
pickle.dump([X_risco_credito, y_risco_credito], f)


from sklearn.naive_bayes import GaussianNB

naive_risco_credito = GaussianNB()
naive_risco_credito.fit(X_risco_credito,y_risco_credito)

previsao = naive_risco_credito.predict([[0,0,1,2],[2,0,0,0]])


#Base credit data
import pickle
file = open('./Bases de dados/credit.pkl', 'rb')
X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = pickle.load(file)

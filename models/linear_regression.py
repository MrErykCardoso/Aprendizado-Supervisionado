#Modelo que aprende uma relação básica entre números

from sklearn.linear_model import LinearRegression
import numpy as np

#Dados de treino
X = np.array([[1], [2], [3], [4], [5]]) #Features
y = np.array([2, 4, 6, 8, 10]) #Labels

#Criação do modelo
model = LinearRegression()

#Treinamento
model.fit(X, y)

#Teste com novos valores
X_novo = np.array([[6], [7], [8], [9], [10]])
y_pred = model.predict(X_novo)

print(f"Respostas do teste:\n {y_pred}")

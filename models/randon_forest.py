#Modelo de classificação de dados com base no método de random forest
#Especificação: classificação de flores e teste de acurácia usando o dataset Iris

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Carregamento de dados
iris = load_iris()
X, y = iris.data, iris.target

#Divisão do dataset entre treino e teste
Xtrain, Xtest, yTrain, yTest = train_test_split(X, y, test_size = 0.3, random_state = 42)

#Modelo 1: árvore de decisão
tree = DecisionTreeClassifier(max_depth = 4)
tree.fit(Xtrain, yTrain)
treePred = tree.predict(Xtest)
print(f"\nResultado do teste da árvore:\n{treePred}")

#Modelo 2: Random forest (bagging - bootstrap aggregation)
forest = RandomForestClassifier(n_estimators = 50, random_state = 42)
forest.fit(Xtrain, yTrain)
forestPred = forest.predict(Xtest)
print(f"\nResultado do teste da floresta:\n{forestPred}")

#Comprando a acurácia
accTree = accuracy_score(yTest, treePred)
accForest = accuracy_score(yTest, forestPred)

print(f"Acurácia da árvore: {accTree:.2f}")
print(f"Acurácia da floresta: {accForest:.2f}")
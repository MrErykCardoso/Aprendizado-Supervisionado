#Modelo de teste e comparação de acurácia entre Rando Forest e Extra Trees
#Especificação: Classificação de flores com base no banco de dados iris

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score

# Carregamento de dados
iris = load_iris()
X, y = iris.data, iris.target

# Divide treino e teste 64% - 36%
Xtrain, Xtest, yTrain, yTest = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Random Forest
forest = RandomForestClassifier(n_estimators = 100, random_state = 42)
forest.fit(Xtrain, yTrain)
forestPred = forest.predict(Xtest)
forestAcc = accuracy_score(yTest, forestPred)

# Extra Trees
extra = ExtraTreesClassifier(n_estimators = 100, random_state = 42)
extra.fit(Xtrain, yTrain)
extraPred = extra.predict(Xtest)
extraAcc = accuracy_score(yTest, extraPred)

print(f"\nAcurácia do modelo Random forest: {forestAcc:.2f}")
print(f"\nAcurácia do modelo Extra Trees: {extraAcc:.2f}")
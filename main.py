import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import setup
import doMath

valData = pd.read_csv('modeling.csv')
testData = pd.read_csv('test.csv')
featureList = ['Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare', 'Embarked']

encodeList = [
    ['Sex', 'male', 0],
    ['Sex', 'female', 1],
    ['Embarked', 'S', 0],
    ['Embarked', 'C', 1],
    ['Embarked', 'Q', 2],
]

for change in encodeList: #Replacements based on the encodeList above
    valData[change[0]].replace(change[1], change[2], inplace=True)
    testData[change[0]].replace(change[1], change[2], inplace=True)

trainX = valData[featureList]
trainY = valData.Survived
valX = testData[featureList] #TODO: train_test_split???

TitanicModel = setup.initialize(featureList, trainX, trainY)
#TitanicModel = setup.optimize(TitanicModel, testX, testY, valX)
doMath.getMAE(TitanicModel, trainX, trainY)

predictions = TitanicModel.predict(valX)
predictions = predictions >= 0.5

tableOutput = setup.formatSystem(predictions, testData.at[0, 'PassengerId'])
setup.spitOut(tableOutput)
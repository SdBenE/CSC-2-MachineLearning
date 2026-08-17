import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
#from sklearn.model_selection import train_test_split
import doMath
import csv

def initialize(featureList, x, y):
    titanicData = pd.read_csv('modeling.csv')
        
    #print(titanicData.columns)
    #print(titanicData)

    #print(titanicData['Sex'])


    #MANUAL NUMERICAL CODING 

    encodeList = [
        ['Sex', 'male', 0],
        ['Sex', 'female', 1],
        ['Embarked', 'S', 0],
        ['Embarked', 'C', 1],
        ['Embarked', 'Q', 2],
    ]

    for change in encodeList:
        titanicData[change[0]].replace(change[1], change[2], inplace=True)

    titanicModel = RandomForestRegressor(max_leaf_nodes=700)
    titanicModel.fit(x, y)

    predicitons = titanicModel.predict(x)
    predicitons = (predicitons >= 0.5)
    print(predicitons)

    return titanicModel

#def optimize(testX, testY, valX, valY): TODO MAYBE???

def formatSystem(predictions, idArg):

    tableArg = pd.DataFrame({'PassengerId' : range(idArg, len(predictions) + idArg)})
    tableArg['Survived'] = predictions

    print(tableArg)
    tableArg['Survived'] = tableArg['Survived'].astype(int)
    print(tableArg)

    return tableArg

def spitOut(tableArg):
    tableArg.to_csv('predictions.csv', index=False)
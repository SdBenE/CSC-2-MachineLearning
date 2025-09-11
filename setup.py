import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
#from sklearn.model_selection import train_test_split
import doMath

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

    for change in encodeList: #Replacements based on the encodeList above
        titanicData[change[0]].replace(change[1], change[2], inplace=True)

    #titanicData['Sex'].replace('male', 1, inplace=True) #changes the data to be modelable by the program
    #titanicData['Sex'].replace('female', 0, inplace=True)

    #titanicData['Embarked'].replace('S', 0, inplace=True)


    #'Cabin',
    print(x)
    titanicModel = RandomForestRegressor()

    titanicModel.fit(x, y)

    predicitons = titanicModel.predict(x)
    predicitons = (predicitons >= 0.5)
    print(predicitons)

    return titanicModel

def optimize(testX, testY, valX, valY):
    bestMAE = 1
    locBestMAE = 0
    for maxLeaves in range(0,500):
        mae = doMath.getMae(maxLeaves, valX, valY, testX, testY)
        print(f"MAE = {mae} with maxLeaves = {maxLeaves}")\
        
        if mae <= bestMAE:
            locBestMAE = maxLeaves
            print(f"New best found with {maxLeaves} leaves! MAE = {mae}")
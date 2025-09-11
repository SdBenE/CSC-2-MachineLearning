import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
#from sklearn.model_selection import train_test_split

titanicData = pd.read_csv('..\Titantic Interpreting Mortality Exposition (TIME)\modeling.csv')

i = 0

    
print(titanicData.columns)
print(titanicData)

print(titanicData['Sex'])

y = titanicData.Survived

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


featureList = ['Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare', 'Embarked']

#'Cabin',

x = titanicData[featureList]
print(x)
titanicModel = RandomForestRegressor()

titanicModel.fit(x, y)

predicitons = titanicModel.predict(x)
predicitons = (predicitons >= 0.5)
print(predicitons)
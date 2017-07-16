from sklearn import tree
import pandas as pd

mydata = pd.read_csv('train.csv')
dataset = mydata
dataset.target = mydata["Survived"]

dataset.data = mydata.drop(['PassengerId', 'Survived', 'Name', 'Sex', 'Age', 'Ticket', 'Cabin', 'Embarked'], axis =1)


clf = tree.DecisionTreeClassifier()
clf = clf.fit(dataset.data, dataset.target)

testdata = pd.read_csv('test.csv')
dataset2 = testdata
dataset2.data = testdata.drop(['PassengerId', 'Name', 'Sex', 'Age', 'Ticket', 'Cabin', 'Embarked'], axis =1)

c= clf.predict(dataset2.data)
output = pd.DataFrame(c, testdata['PassengerId'])
output.to_csv('out.csv')

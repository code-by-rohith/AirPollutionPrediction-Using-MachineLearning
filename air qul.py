# Load libraries
import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

sensor_data = pandas.read_csv("sensor_data.csv")
quality_data = pandas.read_csv("quality_control_data.csv")

rawdataset = sensor_data.merge(quality_data, on="prod_id")

print(rawdataset)

dataset = rawdataset.drop(columns='prod_id')
print(dataset)

# quality distribution
print(dataset.groupby('quality').size())

# box and whisker plots to show data distribution
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# check the histograms
dataset.hist()
plt.show()



# Split-out validation dataset
array = dataset.values
X = array[:,0:3]
Y = array[:,3]
validation_size = 0.20
seed = 8 #parameter 
#Data Split
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='lbfgs')))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10)#cross validation evaluate machine learning models
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare Algorithms
fig = plt.figure()
fig.suptitle('Comparison of ML Models')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

# Make predictions on validation dataset
CART = DecisionTreeClassifier()
CART.fit(X_train, Y_train)
predictions = CART.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

#New input Values
testWeight =  float(input("Enter The Testweight:"))
testHumidity =  float(input("Enter The TestHumidity:"))
testTemperature = float(input("Enter The testTemperture:"))
testPrediction = CART.predict([[testWeight,testHumidity,testTemperature]])
print("Air Quality:",testPrediction)

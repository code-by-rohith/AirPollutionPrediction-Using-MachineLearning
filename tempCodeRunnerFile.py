# Load libraries
import pandas
import numpy as np
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


# Split-out validation dataset
array = dataset.values
X = array[:,0:3]
Y = array[:,3]
validation_size = 0.20
seed = 8 #parameter 
#Data Split
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
print("X_train",X_train.shape)
print("X_validation",X_validation.shape)
print("Y_train",Y_train.shape)
print("Y_validation",Y_validation.shape)


from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(X_train, Y_train)  #train the data
y_pred=NB.predict(X_validation)
##print(y_pred)
##print(y_test)
print('Naive Bayes ACCURACY is', accuracy_score(Y_validation,y_pred))

############################# RandomForestClassifier  #######################
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train, Y_train)  #train the data
y_pred_rf=clf.predict(X_validation)
##print(y_pred)
##print(y_test)
print('Random Forest ACCURACY is', accuracy_score(Y_validation,y_pred_rf))

from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(random_state=0)
DT.fit(X_train, Y_train)  #train the data
y_pred_DT=DT.predict(X_validation)
##print(y_pred)
##print(y_test)
print('DecisionTree ACCURACY is', accuracy_score(Y_validation,y_pred_DT))

from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier()
KNN.fit(X_train, Y_train)  #train the data
y_pred_KNN=KNN.predict(X_validation)
##print(y_pred)
##print(y_test)
print('KNeighborsClassifier ACCURACY is', accuracy_score(Y_validation,y_pred_KNN))


from sklearn import svm
svc=clf = svm.SVC()
svc.fit(X_train, Y_train)  #train the data
y_pred_svc=svc.predict(X_validation)
##print(y_pred)
##print(y_test)
print('Support Vector Machine ACCURACY is', accuracy_score(Y_validation,y_pred_svc))



from flask import Flask, request, jsonify, render_template
import pickle
import warnings
warnings.filterwarnings("ignore")
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
   
    prediction = DT.predict(final_features)
##    pred = None
##    if prediction == 0:
##        pred = "The Quality of the Air is Good"
##    else:
##        pred = "The Quality of the Air is Bad"
##    output = pred
    return render_template('index.html', prediction_text='The Quality of the Air is - {}'.format(prediction))

if __name__ == "__main__":
    app.run()

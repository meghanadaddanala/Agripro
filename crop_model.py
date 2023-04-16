#importing libraries
import pandas as pd
import numpy as np
import pickle

#reading the data
data = pd.read_csv("Crop_recommendation.csv")

#data preprocessing
data['label'].unique()
data['label'].replace(['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango', 'grapes','watermelon', 'muskmelon', 'apple', 'orange', 'papaya', 'coconut', 'cotton','jute', 'coffee'], [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22], inplace=True)

from sklearn.model_selection import train_test_split
x = data.drop('label', axis=1)
y = data['label']

#splitting the dataset
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.1, random_state=72)

#normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(xtrain)

X_test_scaled = scaler.transform(xtest)
# training the model
#random forest classifier
from sklearn.ensemble import RandomForestClassifier

model =RandomForestClassifier()

#train our model
model.fit(X_train_scaled,ytrain)

#testing model
ypred=model.predict(X_test_scaled)

#accuracies of the model - progress report
print('Training Accuracy :', model.score(X_train_scaled, ytrain))
print('Testing Accuracy :', model.score(X_test_scaled, ytest) )

#testing
input_N=[float(x) for x in "90 42 43 20.87974371 82.00274423 6.502985292000001 202.9355362".split(' ')]
final=[np.array(input_N)]
print(final)
prediction=model.predict(scaler.transform(final))
print(prediction)
crops=['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate', 'banana', 'mango', 'grapes','watermelon', 'muskmelon', 'apple', 'orange', 'papaya', 'coconut', 'cotton','jute', 'coffee']
print(crops[prediction[0]-1])
#print(prediction)


#pickling
pickle.dump(model, open('model/crop_model.pkl', 'wb'))
model=pickle.load(open('model/crop_model.pkl', 'rb'))
print("SUcess loaded")

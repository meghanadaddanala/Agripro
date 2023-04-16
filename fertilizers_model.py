# loading the  dataset
import pickle

import pandas as pd
data = pd.read_csv("finally_fertilizerdataset_2.csv",encoding='unicode_escape')

data['crop'].replace(['rice', 'maize','chickpea','kidneybeans','pigeonpeas',
       'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
       'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
       'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee'], [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21], inplace=True)
data

data['fertilizer'].replace(['Manure', 'Urea', 'phosphorus fertilize', 'Erwon NPK 30:10:10','Nitrogen', 'Phosphorus', 'Complex NPK fertilizers', 'Fertiberia','Potassium', 'Nitrogen fertilizers', 'K2O', 'P2O5', 'N-P-K','TrustBasket based Fertilizer', 'IFFCO fertilizer',"Jobe's Fruit fertilizer", 'Down to earth Organic fertilizer','Hydrated Lime', 'liquid seaweed', 'Calciun Nitrate', 'Boron','Azospirillum', 'Phosphobacteria', 'Khandelwal Bio Fertilizer','Apple Calcium Chloride Agriculture & Fertilizer Powder','Phosphate', "Jobe's Fruit & Citrus Fertilizer Spikes",'Manganese sulfate 200 gm', 'Calcium nitrate 250 gm', 'EcoScraps','Poultry manure', 'Zinc Sulphate 50 kg.', 'SSP 375 kg.','Urea 15kg', 'MgSO4', 'potash', 'Di-ammonium Phosphate'],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,], inplace=True)
print(data)
from sklearn.model_selection import train_test_split, cross_val_score

x = data.drop('fertilizer', axis=1)# N P K temperature humidity ph rainfall
y = data['fertilizer']#label

#splitting the dataset
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.5, random_state=42)

#normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(xtrain)

# we must apply the scaling to the test set as well that we are computing for the training set
X_test_scaled = scaler.transform(xtest)


#svm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
clf =LogisticRegression(penalty='l2',C=0.001,random_state=0)
clf .fit(X_train_scaled,ytrain)

ypred=clf .predict(X_test_scaled)
print("Logistic Regression Accuracy: ",clf.score(scaler.transform(x),y))



#Evaluating the model

output={'y_pred':ypred,'y_actual':ytest}
output=pd.DataFrame(output)
print(output)

#accuracies of the model
print('Training Accuracy :', clf .score(X_train_scaled, ytrain))
print('Testing Accuracy :', clf .score(X_test_scaled, ytest) )

#Model 2--> Random Forest Algo
#random forest classifier
from sklearn.ensemble import RandomForestClassifier

model =RandomForestClassifier(max_depth=15,random_state=42)
model.fit(X_train_scaled,ytrain)

ypred=model.predict(X_test_scaled)
#accuracies of the model
c=model.score(scaler.fit_transform(x),y)
print('Training Accuracy :', model.score(X_train_scaled, ytrain))
print('Testing Accuracy :', model.score(X_test_scaled, ytest) )
print('Overall Accuracy :',c)

#pickling
pickle.dump(model, open('model/fertilizer_model.pkl', 'wb'))
model1=pickle.load(open('model/fertilizer_model.pkl', 'rb'))
print("SUcess loaded")
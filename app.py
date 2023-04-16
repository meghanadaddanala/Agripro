#Coding a Server - Flask

from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import requests as reqs


app = Flask(__name__) #configuring the server

model=pickle.load(open('model/crop_model.pkl','rb')) #crop recommendation model


fertilizer_clf=pickle.load(open('model/fertilizer_model.pkl','rb'))


import pandas as pd
import numpy as np
import pickle

#reading the data

#splitting the dataset

#serving the starting web page to the browser
@app.route('/')
def hello_world():
    return render_template("index.html")

#There are HTTP Requests - POST & GET
#Server to serve inputs from Web Page (Frontend) to Backend (ML model) - uses POST
#Server to serve back the output from backend(ML model) to Frontend (Web page) - uses GET
@app.route('/crop',methods=['POST','GET']) #server kind of syntax to call this method
def crop():
    features = [(x) for x in request.form.values()]
    print(features)
    reqs.post(
        'https://ap-south-1.aws.data.mongodb-api.com/app/application-0-wjtvm/endpoint/Crop_RegisterPost',
        data={"name": features[0],
              "mobile":features[1],
              "crop":features[2],
              "date":features[3],
              "state":features[4],
              "district":features[5],
              "village":features[6],
              "location":features[7],
              "status":features[8]})
    print("done")
    return render_template('farmer_crop_register.html',
                           pred='Successfully Registered')

@app.route('/predict',methods=['POST','GET'])
def predict():
    features = [(x) for x in request.form.values()]
    final_features = [np.array(features)]
    print(final_features)
    c = final_features[0]
    print(c)
    import requests
    from datetime import datetime
    import pytz
    city = c[6]
    print(city)
    date_time = datetime.now().strftime("%d %b %Y | %I:%M:%S %p")
    print(date_time)
    api_key1 = 'ad62ecebb7931902c9fdbfefb78f3277'

    url = 'http://api.openweathermap.org/data/2.5/weather?q={}&appid={}&units=metric'.format(city, api_key1)
    res = requests.get(url)
    data = res.json()
    print(data)

    latitude = data['coord']['lat']
    longitude = data['coord']['lon']
    print('latitude :', latitude)
    print('longitude :', longitude)
    # getting the main dict block
    main = data['main']
    wind = data['wind']
    # getting temperature
    temperature = main['temp']
    # getting the humidity
    humidity = main['humidity']


    # weather report
    report = data['weather']
    print(f"Temperature : {temperature}°C")
    print(f"Humidity : {humidity}")

    print(f"Weather Report : {report[0]['description']}")

    float_features = [(x) for x in request.form.values()]
    print("Values:")
    inputs_final=[1,2,3,4,5,6,7]
    inputs_final[0]=float_features[0]
    inputs_final[1] = float_features[1]
    inputs_final[2] = float_features[2]
    inputs_final[3] = temperature
    inputs_final[4] = humidity
    inputs_final[5] = float_features[3]
    inputs_final[6] = float_features[7]
    print(inputs_final)

    data = pd.read_csv("Crop_recommendation.csv")
    from sklearn.model_selection import train_test_split
    x = data.drop('label', axis=1)
    y = data['label']
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.1, random_state=72)
    #input_N = [float(x) for x in "90 85 60 74 78 69 69".split(' ')]
    final = [np.array(inputs_final)] #converting list to array
    print(final)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(xtrain)   #to know the scale
    prediction = model.predict(scaler.transform(final))
    print(prediction)
    crops = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'mungbean', 'blackgram', 'lentil',
             'pomegranate', 'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya',
             'coconut', 'cotton', 'jute', 'coffee']
    print(crops[prediction[0]-1])

    #output=crops[prediction[0] - 1]
    # print(prediction)
    #Fertilizers Prediction
    fertilizers_input=[1,2,3,4,5,6,7,8]
    fertilizers_input[0] = float_features[0]
    fertilizers_input[1] = float_features[1]
    fertilizers_input[2] = float_features[2]
    fertilizers_input[3] = temperature
    fertilizers_input[4] = humidity
    fertilizers_input[5] = float_features[3]
    fertilizers_input[6] = float_features[7]
    fertilizers_input[7] = prediction-1
    print(fertilizers_input)

    data = pd.read_csv("finally_fertilizerdataset_2.csv", encoding='unicode_escape')

    data['crop'].replace(['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
                          'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
                          'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
                          'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee'],
                         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], inplace=True)
    data

    data['fertilizer'].replace(
        ['Manure', 'Urea', 'phosphorus fertilize', 'Erwon NPK 30:10:10', 'Nitrogen', 'Phosphorus',
         'Complex NPK fertilizers', 'Fertiberia', 'Potassium', 'Nitrogen fertilizers', 'K2O', 'P2O5', 'N-P-K',
         'TrustBasket based Fertilizer', 'IFFCO fertilizer', "Jobe's Fruit fertilizer",
         'Down to earth Organic fertilizer', 'Hydrated Lime', 'liquid seaweed', 'Calciun Nitrate', 'Boron',
         'Azospirillum', 'Phosphobacteria', 'Khandelwal Bio Fertilizer',
         'Apple Calcium Chloride Agriculture & Fertilizer Powder', 'Phosphate',
         "Jobe's Fruit & Citrus Fertilizer Spikes", 'Manganese sulfate 200 gm', 'Calcium nitrate 250 gm', 'EcoScraps',
         'Poultry manure', 'Zinc Sulphate 50 kg.', 'SSP 375 kg.', 'Urea 15kg', 'MgSO4', 'potash',
         'Di-ammonium Phosphate'],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
         31, 32, 33, 34, 35, 36, 37, ], inplace=True)

    from sklearn.model_selection import train_test_split, cross_val_score

    x = data.drop('fertilizer', axis=1)  # N P K temperature humidity ph rainfall
    y = data['fertilizer']  # label

    # splitting the dataset
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.5, random_state=42)

    # normalization
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(xtrain)

    # we must apply the scaling to the test set as well that we are computing for the training set
    X_test_scaled = scaler.transform(xtest)
    final1 = [np.array(fertilizers_input)]
    p = fertilizer_clf.predict(scaler.transform(final1))
    print(p)

    f=['Manure', 'Urea', 'phosphorus fertilize', 'Erwon NPK 30:10:10', 'Nitrogen', 'Phosphorus',
         'Complex NPK fertilizers', 'Fertiberia', 'Potassium', 'Nitrogen fertilizers', 'K2O', 'P2O5', 'N-P-K',
         'TrustBasket based Fertilizer', 'IFFCO fertilizer', "Jobe's Fruit fertilizer",
         'Down to earth Organic fertilizer', 'Hydrated Lime', 'liquid seaweed', 'Calciun Nitrate', 'Boron',
         'Azospirillum', 'Phosphobacteria', 'Khandelwal Bio Fertilizer',
         'Apple Calcium Chloride Agriculture & Fertilizer Powder', 'Phosphate',
         "Jobe's Fruit & Citrus Fertilizer Spikes", 'Manganese sulfate 200 gm', 'Calcium nitrate 250 gm', 'EcoScraps',
         'Poultry manure', 'Zinc Sulphate 50 kg.', 'SSP 375 kg.', 'Urea 15kg', 'MgSO4', 'potash',
         'Di-ammonium Phosphate']
    print(f[p[0]+1])

    return render_template('index.html',
                           pred='Recommended Crop: {}'.format(crops[prediction[0]-1]),inp='Fertilizer that needed to be used: {}'.format(f[p[0]]))

if __name__ == '__main__':
    app.run(debug=True)



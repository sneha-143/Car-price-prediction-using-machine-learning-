from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('C:\\Users\\A SNEHA\\OneDrive\\Desktop\\ml projects\\LinearRegressionModel.pkl', 'rb'))
car=pd.read_csv('C:\\Users\\A SNEHA\\OneDrive\\Desktop\\ml projects\\car_price (1).csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    car_ID = sorted(car['car_ID'].unique())
    symboling = sorted(car['symboling'].unique())
    CarName = sorted(car['CarName'].unique())
    fueltype = car['fueltype'].unique()
    aspiration = car['aspiration'].unique()
    doornumber = sorted(car['doornumber'].unique())
    carbody = sorted(car['carbody'].unique())
    drivewheel = sorted(car['drivewheel'].unique())
    enginelocation = sorted(car['enginelocation'].unique())
    wheelbase = sorted(car['wheelbase'].unique())
    carlength = sorted(car['carlength'].unique())
    carwidth = sorted(car['carwidth'].unique())
    carheight = sorted(car['carheight'].unique())
    curbweight = sorted(car['curbweight'].unique())
    enginetype = sorted(car['enginetype'].unique())
    cylindernumber = sorted(car['cylindernumber'].unique())
    enginesize = sorted(car['enginesize'].unique())
    fuelsystem = sorted(car['fuelsystem'].unique())
    boreratio = sorted(car['boreratio'].unique())
    stroke = sorted(car['stroke'].unique())
    compressionratio = sorted(car['compressionratio'].unique())
    horsepower = sorted(car['horsepower'].unique())
    peakrpm = sorted(car['peakrpm'].unique())
    citympg = sorted(car['citympg'].unique())
    highwaympg = sorted(car['highwaympg'].unique())

    car_ID.insert(0, 'Select car_ID')
    return render_template('index.html', car_ID=car_ID, symboling=symboling, CarName=CarName, fueltype=fueltype, aspiration=aspiration, doornumber=doornumber, carbody=carbody, drivewheel=drivewheel, enginelocation=enginelocation, wheelbase=wheelbase, carlength=carlength, carwidth=carwidth, carheight=carheight, curbweight=curbweight, enginetype=enginetype, cylindernumber=cylindernumber, enginesize=enginesize, fuelsystem=fuelsystem, boreratio=boreratio, stroke=stroke, compressionratio=compressionratio, horsepower=horsepower, peakrpm=peakrpm, citympg=citympg, highwaympg=highwaympg)



@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    car_ID = request.form.get('car_ID')
    symboling = request.form.get('symboling')
    CarName = request.form.get('CarName')
    fueltype = request.form.get('fueltype')
    aspiration = request.form.get('aspiration')
    doornumber = request.form.get('doornumber')
    carbody = request.form.get('carbody')
    drivewheel = request.form.get('drivewheel')
    enginelocation = request.form.get('enginelocation')
    wheelbase = request.form.get('wheelbase')
    carlength = request.form.get('carlength')
    carwidth = request.form.get('carwidth')
    carheight = request.form.get('carheight')
    curbweight = request.form.get('curbweight')
    enginetype = request.form.get('enginetype')
    cylindernumber = request.form.get('cylindernumber')
    enginesize = request.form.get('enginesize')
    fuelsystem = request.form.get('fuelsystem')
    boreratio = request.form.get('boreratio')
    stroke = request.form.get('stroke')
    compressionratio = request.form.get('compressionratio')
    horsepower = request.form.get('horsepower')
    peakrpm = request.form.get('peakrpm')
    citympg = request.form.get('citympg')
    highwaympg = request.form.get('highwaympg')

    input_data = {
        'CarName': CarName,
        'fueltype': fueltype,
        'aspiration': aspiration,
        'doornumber': doornumber,
        'carbody': carbody,
        'drivewheel': drivewheel,
        'enginelocation': enginelocation,
        'wheelbase': wheelbase,
        'carlength': carlength,
        'carwidth': carwidth,
        'carheight': carheight,
        'enginetype': enginetype,
        'cylindernumber': cylindernumber,
        'enginesize': enginesize,
        'fuelsystem': fuelsystem,
        'compressionratio': compressionratio,
        'horsepower': horsepower,
        'peakrpm': peakrpm,
        'citympg': citympg,
    }

    prediction=model.predict(pd.DataFrame(columns=['car_ID', 'symboling', 'CarName', 'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation', 'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'fuelsystem', 'boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg'],
                              data=np.array([car_ID, symboling, CarName, fueltype, aspiration, doornumber, carbody, drivewheel, enginelocation, wheelbase, carlength, carwidth, carheight, curbweight, enginetype, cylindernumber, enginesize, fuelsystem, boreratio, stroke, compressionratio, horsepower, peakrpm, citympg, highwaympg]).reshape(1, 25)))
    print(prediction)

    return str(np.round(prediction[0],2))



if __name__ == '__main__':
    app.run()
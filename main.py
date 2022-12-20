from flask import Flask, request, jsonify , render_template
import numpy as np
import pickle


app = Flask(__name__)

model_car = pickle.load(open('random_forest_regression_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/cars', methods=['POST'])
def car():
    Fuel_Type_Diesel = 0
    if request.method == 'POST':
        Year = int(request.form.get('Year'))
        Present_Price = float(request.form.get('Present_Price'))
        Kms_Driven = int(request.form.get('Kms_Driven'))
        Kms_Driven2 = np.log(Kms_Driven)
        Owner = int(request.form.get('Owner'))
        Fuel_Type_Petrol = request.form.get('Fuel_Type_Petrol')

        if (Fuel_Type_Petrol == 'Petrol'):
            Fuel_Type_Petrol = 1
            Fuel_Type_Diesel = 0
        else:
            Fuel_Type_Petrol = 0
            Fuel_Type_Diesel = 1

        Year = 2022 - Year

        Seller_Type_Individual = request.form.get('Seller_Type_Individual')

        if (Seller_Type_Individual == 'Individual'):
            Seller_Type_Individual = 1
        else:
            Seller_Type_Individual = 0

        Transmission_Mannual = request.form.get('Transmission_Mannual')

        if (Transmission_Mannual == 'Mannual'):
            Transmission_Mannual = 1
        else:
            Transmission_Mannual = 0

        # result = {'Present_Price': Present_Price, 'Kms_Driven': Kms_Driven2, 'Owner': Owner, 'Year': Year,
        #          "Fuel_Type_Petrol": Fuel_Type_Petrol,
        #           "Seller_Type_Individual": Seller_Type_Individual,
        #           "Transmission_Mannual": Transmission_Mannual,
        #           }

        input_query = np.array([[Present_Price, Kms_Driven2, Owner, Year, Fuel_Type_Diesel, Fuel_Type_Petrol,
                                         Seller_Type_Individual, Transmission_Mannual]])
        prediction = model_car.predict(input_query)[0]
        result = round(prediction ,2 )
        return jsonify({'Predicted_Price': result})


if __name__ == '__main__':
    app.run(debug=True)

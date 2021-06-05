from flask import Flask,render_template,request
import pickle
import numpy as np
from crop_descritption_map import cd


app = Flask(__name__)


@app.route("/")
def home():
    return render_template('index.html')

@app.route('/form')
def formpage():
    return render_template('form.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/predict',methods=['POST'])
def predict():
    N = request.form['N']
    print(N)
    P = request.form['P']
    K = request.form['K']
    T = request.form['T']
    H = request.form['H']
    P = request.form['P']
    R = request.form['R']

    Soil_composition_list = np.array([N,P,K,T,H,P,R]).reshape(1,7)
    print(Soil_composition_list)
    loaded_model = pickle.load(open('Crop_recommendation_model.sav', 'rb'))
    crop = loaded_model.predict(Soil_composition_list)
    print(crop)

    if crop == 'rice':
        return render_template('rice.html')
    
    elif crop == 'maize':
        return render_template('maize.html')
    
    else:
        predicted_crop = crop[0]
        crop_discription = cd[crop[0]]['crop_dis']
        crop_link = cd[crop[0]]['crop_link']
        crop_img = cd[crop[0]]['crop_img']


        return render_template('prediction.html', predicted_crop = predicted_crop , crop_discription = crop_discription , crop_link = crop_link , crop_img = crop_img )



if __name__ == "__main__":
 app.run(host='0.0.0.0', port=5000,debug=True)
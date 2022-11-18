from logging import debug
from flask import Flask, render_template, request 
import utils  
from utils import preprocessdata 
from constants import *

app = Flask(__name__) 

@app.route('/', methods=['GET']) 
def home(): 
    return render_template('index.html', feature_set_1=feature_set_1, feature_set_2=feature_set_2, feature_set_3=feature_set_3, feature_set_4=feature_set_4) 

@app.route('/predict/', methods=['POST'])
def predict():  
    # print(request.form)

    # Children = request.form.get('Children')
    # Income = request.form.get('Income')
    # Workphone = request.form.get('Workphone')
    # Mobile = request.form.get('Mobile')
    # Email = request.form.get('Email')
    # Family = request.form.get('Family')
    # Age = request.form.get('Age')
    # Emp_months = request.form.get('Emp_months')
    # Female = request.form.get('Female')
    # Male = request.form.get('Male')
    # CNo = request.form.get('CNo')
    # CYes = request.form.get('CYes')
    # RealtyNo = request.form.get('RealtyNo')
    # RealtyYes = request.form.get('RealtyYes')
    # Associate = request.form.get('Associate')
    # Pensioner = request.form.get('Pensioner')
    # State = request.form.get('State')
    # Working = request.form.get('Working')
    # AcademicDegree = request.form.get('AcademicDegree')
    # HighEd = request.form.get('HighEd')
    # Incomplete = request.form.get('Incomplete')
    # LowerSecond = request.form.get('LowerSecond')
    # Special = request.form.get('Special')
    # Civil = request.form.get('Civil')
    # Married = request.form.get('Married')
    # Separated = request.form.get('Separated')
    # Single = request.form.get('Single')
    # Widow = request.form.get('Widow')
    # Coop = request.form.get('Coop')  
    # House = request.form.get('House')  
    # Municipal = request.form.get('Municipal') 
    # Office = request.form.get('Office')   
    # Rented = request.form.get('Rented')   
    # Parents = request.form.get('Parents')   
    # Accountants = request.form.get('Accountants')  
    # Cleaning = request.form.get('Cleaning')  
    # Cooking = request.form.get('Cooking')  
    # Core = request.form.get('Core') 
    # Drivers = request.form.get('Drivers')   
    # HR = request.form.get('HR')   
    # Tech = request.form.get('Tech')   
    # IT = request.form.get('IT')  
    # Laborers = request.form.get('Laborers')  
    # LowSkillLaborers = request.form.get('LowSkillLaborers')  
    # Skill = request.form.get('Skill')  
    # Managers = request.form.get('Managers') 
    # Medical = request.form.get('Medical')   
    # Private = request.form.get('Private')   
    # Agents = request.form.get('Agents')   
    # Secretaries = request.form.get('Secretaries')  
    # Security = request.form.get('Security')   
    # WB = request.form.get('WB')
    
    features = request.form.to_dict()
    for k, v in features.items():
        features[k] = float(v)

    prediction = utils.preprocessdata(features)

    # return str(prediction)
    return render_template('predict.html', prediction=prediction) 

if __name__ == '__main__': 
    app.run(host='0.0.0.0', debug=True) 

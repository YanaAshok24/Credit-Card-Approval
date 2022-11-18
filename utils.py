import numpy as np 
import pandas as pd
import joblib 
import xgboost as xgb


trained_model = xgb.XGBClassifier()
trained_model.load_model('./static/xgboost.json')

# def preprocessdata(Children, Income, Workphone, Mobile, Email, Family, Age, Emp_months, Female, Male, CNo, CYes, RealtyNo, RealtyYes,
#       Associate,Pensioner, State, Working, AcademicDegree, HighEd, Incomplete, LowerSecond,Special, Civil, Married, Separated, Single, Widow,
#       Coop, House, Municipal, Office, Rented, Parents,
#       Accountatnts, Cleaning, Cooking, Core, Drivers, HR, Tech, IT, Laborers, LowSkillLaborers, Skill, Managers, Medical, Private, Agents,
#       Secretaries, Security, WB):
#     test_data = [[Children, Income, Workphone, Mobile, Email, Family, Age, Emp_months, Female, Male, CNo, CYes, RealtyNo, RealtyYes,
#       Associate,Pensioner, State,Working, AcademicDegree, HighEd, Incomplete, LowerSecond,Special, Civil, Married, Separated, Single, Widow,
#       Coop, House, Municipal, Office, Rented, Parents,
#       Accountatnts, Cleaning, Cooking, Core, Drivers, HR, Tech, IT, Laborers, LowSkillLaborers, Skill, Managers, Medical, Private, Agents,
#       Secretaries, Security, WB]]  
def preprocessdata(features):
    test_data = pd.DataFrame(features, index=[0])
    # trained_model = joblib.load("./static/xgboost.pkl") 
    prediction = trained_model.predict(test_data) 

    return prediction 
# '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', None, '0', '0', '0', '0', '0', None, '0', '0', '0', '0', '0', None, '0', '0', None, '0', '0', '0', '0', '0', '0', '0'

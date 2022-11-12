import numpy as np 
import joblib 


def preprocessdata(Children, Income, Workphone, Mobile, Email, Family, Age, Emp_months, Female, Male, CNo, CYes, RealtyNo, RealtyYes,
      Associate,Pensioner, State,Working, HighEd, Incomplete, LowerSecond,Special, Civil, Married, Separated, Single, Widow,
      Coop, House, Municipal, Office, Rented, Parents,
      Accountatnts, Cleaning, Cooking, Core, Drivers, HR, Tech, IT, Laborers, Skill, Managers, Medical, Private, Agents,
      Secretaries, Security, WB):
    test_data = [[Children, Income, Workphone, Mobile, Email, Family, Age, Emp_months, Female, Male, CNo, CYes, RealtyNo, RealtyYes,
      Associate,Pensioner, State,Working, HighEd, Incomplete, LowerSecond,Special, Civil, Married, Separated, Single, Widow,
      Coop, House, Municipal, Office, Rented, Parents,
      Accountatnts, Cleaning, Cooking, Core, Drivers, HR, Tech, IT, Laborers, Skill, Managers, Medical, Private, Agents,
      Secretaries, Security, WB] ]  
    trained_model = joblib.load("xgboost.pkl") 
    prediction = trained_model.predict(test_data) 

    return prediction 
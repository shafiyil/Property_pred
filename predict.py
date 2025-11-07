import joblib
import pandas as pd

area = float(input("Enter the area in square feet "))
bedrooms = int(input("Enter the number of bedroom "))
age = int(input("Enter the age of the house in years "))

model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
features_name = joblib.load('features_name.joblib')

input_data = {
    'Area' : area,
    'Bedrooms' : bedrooms,
    'Age' : age
}

X_new = pd.DataFrame([input_data], columns=features_name)

X_new_scaled = scaler.transform(X_new)

predicted_price = model.predict(X_new_scaled)

print(f"Pridected Price of the House is : ${predicted_price[0]:,.2f}")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib


df=pd.read_csv("data/property.csv")
print(df.info())

X=df.drop(columns=["Price"])
features_names = X.columns.to_list()
y=df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_Scaled = scaler.fit_transform(X_train)
X_test_Scaled = scaler.transform(X_test)

#create model

model = LinearRegression()
model.fit(X_train_Scaled, y_train)

# export to joblib

joblib.dump(model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(features_names, 'features_name.joblib')
# #check accuracy

# y_pred=model.predict(X_test_Scaled)
# print("MAE:",mean_absolute_error(y_test, y_pred))
# #print("***************************************************************************************************************")
# #print("Actual price:\n",(y_test))
# #print("***************************************************************************************************************")
# #print("Predected Price:\n",(y_pred))

# Area = float(input("Enter the square feet "))
# Bedrooms = int(input("Enter no of bedrooms "))
# Age = int(input("Enter the age of your property in years "))

# new_data ={
#     'Area' : Area,
#     'Bedrooms' : Bedrooms,
#     'Age' : Age
# }
# new_df = pd.DataFrame([new_data])

# new_of_scaled = scaler.transform(new_df)
# predicted_price = model.predict(new_of_scaled)
# print(f"The Predicted price of property is : ${predicted_price[0]:,.2f}")
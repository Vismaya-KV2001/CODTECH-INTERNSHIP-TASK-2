/* Car Price Prediction using Linear Regression */

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Load the data set
file_path=r"C:\Users\visma\OneDrive\Desktop\car_prices\car_prediction_data.csv"
df = pd.read_csv(file_path)

#Explore the data set
print(df.info())
print(df.describe())
print(df.isnull().sum())  # Check for missing values

#Preprocess the Data
#Feature Engineering
df['Car_Age'] = 2024 - df['Year']
df.drop(['Car_Name', 'Year'], axis=1, inplace=True)  # Drop unnecessary columns

#Encode Categorical Variables
df = pd.get_dummies(df, drop_first=True)

#Check correlation
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

#Split the data set 
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']


#Creating training and test set:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train the linear regression model
#initialize and train
model = LinearRegression()
model.fit(X_train, y_train)

#check the model coefficients
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

#Evaluate model
#Make predictions
y_pred = model.predict(X_test)
 
#Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
 

#Visualize result
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Selling Price")
plt.show()

#Save and test
##############################################################################################################

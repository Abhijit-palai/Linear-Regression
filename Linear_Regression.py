#Linear Regression

#Import and preprocess the dataset

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#  Load the dataset
df = pd.read_csv('Titanic-Dataset.csv')

 #Basic preprocessing
#Drop rows with missing values 
df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
df = df.dropna()

#  Separate features and target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  Fit a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

#  Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R2 Score:", r2)

# Interpret coefficients
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print("\nFeature Coefficients:")
print(coefficients)

# Plot Regression Line for one feature ('Fare')

# For plotting, we pick 'Fare' feature
# Re-train a model using only 'Fare'
X_fare = df[['Fare']]  # Keep as DataFrame
y = df['Survived']

X_train_fare, X_test_fare, y_train_fare, y_test_fare = train_test_split(
    X_fare, y, test_size=0.2, random_state=42
)

model_fare = LinearRegression()
model_fare.fit(X_train_fare, y_train_fare)

# Predict
y_pred_fare = model_fare.predict(X_test_fare)

# Plot
plt.figure(figsize=(8,5))
plt.scatter(X_test_fare, y_test_fare, color='blue', label='Actual data')
plt.plot(X_test_fare, y_pred_fare, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Fare')
plt.ylabel('Survived')
plt.title('Linear Regression: Fare vs Survived')
plt.legend()
plt.show()

# Interpret the coefficient for 'Fare'
print("\nModel trained with only 'Fare' feature:")
print("Coefficient (Slope):", model_fare.coef_[0])
print("Intercept:", model_fare.intercept_)


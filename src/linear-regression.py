# Import libraries & dataset
# import shap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

df = pd.read_csv('./data/dataset-preprocessed.csv')

# Creation of the train & test datasets

X = df.drop(columns=['Price'], axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standaridization of the values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Creation of the model
regression = LinearRegression()
regression.fit(X_train, y_train)

# Prediction using the test set
prediction = regression.predict(X_test)
prediction_train = regression.predict(X_train)

# Checking how well the model is correct
comparison = pd.DataFrame({'Real values':y_test, 'Predicted values':prediction})
print(comparison.head(10).sort_values(by='Real values', ascending=True))

print("MAE - training:", round(mean_absolute_error(y_train, prediction_train),2))
print("MAE - test:", round(mean_absolute_error(y_test, prediction),2))

print("MSE -training:", round(mean_squared_error(y_train, prediction_train),2))
print("MSE - test:", round(mean_squared_error(y_test, prediction),2))

print("RMSE - training:", round(np.sqrt(mean_squared_error(y_train, prediction_train)),2))
print("RMSE - test:", round(np.sqrt(mean_squared_error(y_test, prediction)),2))

print('Score train :', regression.score(X_train, y_train))
print('Score test :', regression.score(X_test, y_test))
print(r2_score(y_test, prediction))


plt.clf()
plt.scatter(x=y_test,y=prediction)
plt.savefig(f"./graphs/prediction-vs-testdata.png")



# Get coefficients
coefficients = regression.coef_

# Display feature importance
features = [f'Feature {i}' for i in range(X.shape[1])]
importance_df = pd.DataFrame({'Feature': features, 'Importance': coefficients})
importance_df.sort_values(by='Importance', ascending=False, inplace=True)

plt.bar(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Feature')
plt.ylabel('Coefficient Value')
plt.title('Feature Importance in Linear Regression')
plt.xticks(rotation=90)    
plt.show()

df.to_csv('./data/comparison.csv')

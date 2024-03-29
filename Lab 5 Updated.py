#!/usr/bin/env python
# coding: utf-8

# # <font color=darkblue> Machine Learning model deployment with Flask framework</font>

# ## <font color=Blue>Used Cars Price Prediction Application</font>

# ### Objective:
# 1. To build a Machine learning regression model to predict the selling price of the used cars based on the different input features like fuel_type, kms_driven, type of transmission etc.
# 2. Deploy the machine learning model with the help of the flask framework.

# ### Dataset Information:
# #### Dataset Source: https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho?select=CAR+DETAILS+FROM+CAR+DEKHO.csv
# This dataset contains information about used cars listed on www.cardekho.com
# - **Car_Name**: Name of the car
# - **Year**: Year of Purchase
# - **Selling Price (target)**: Selling price of the car in lakhs
# - **Present Price**: Present price of the car in lakhs
# - **Kms_Driven**: kilometers driven
# - **Fuel_Type**: Petrol/diesel/CNG
# - **Seller_Type**: Dealer or Indiviual
# - **Transmission**: Manual or Automatic
# - **Owner**: first, second or third owner
# 

# ### 1. Import required libraries

# In[1]:


pip install pandas


# In[1]:


import pandas as pd


# In[9]:


df = pd.read_csv(r'C:\Users\PURNANGSHU ROY\OneDrive\Desktop\car\cardata.csv')


# ### 2. Load the dataset

# In[10]:


import pandas as pd

# Load the dataset
df = pd.read_csv(r'C:\Users\PURNANGSHU ROY\OneDrive\Desktop\car\cardata.csv')

# Display the first few rows of the DataFrame
print(df.head())


# ### 3. Check the shape and basic information of the dataset.

# In[11]:


import pandas as pd

# Load the dataset
df = pd.read_csv(r'C:\Users\PURNANGSHU ROY\OneDrive\Desktop\car\cardata.csv')

# Check the shape of the dataset
print("Shape of the dataset:", df.shape)

# Display basic information about the dataset
print("\nBasic Information about the dataset:")
print(df.info())


# ### 4. Check for the presence of the duplicate records in the dataset? If present drop them

# In[12]:


import pandas as pd

# Load the dataset
df = pd.read_csv(r'C:\Users\PURNANGSHU ROY\OneDrive\Desktop\car\cardata.csv')

# Check for duplicate records
duplicate_rows = df[df.duplicated()]

if not duplicate_rows.empty:
    print("Duplicate records found. Dropping them...")
    # Drop duplicate records
    df.drop_duplicates(inplace=True)
    print("Duplicate records have been dropped.")
else:
    print("No duplicate records found in the dataset.")

# Check the shape of the dataset after dropping duplicates
print("\nShape of the dataset after dropping duplicates:", df.shape)


# ### 5. Drop the columns which you think redundant for the analysis.

# In[14]:


import pandas as pd

# Load the dataset
df = pd.read_csv(r'C:\Users\PURNANGSHU ROY\OneDrive\Desktop\car\cardata.csv')

# List of columns considered redundant for analysis
redundant_columns = ['Seller_Type', 'Transmission', 'Owner']

# Drop the redundant columns
df.drop(columns=redundant_columns, inplace=True)

# Display the first few rows of the DataFrame after dropping columns
print(df.head())


# ### 6. Extract a new feature called 'age_of_the_car' from the feature 'year' and drop the feature year

# In[20]:


import pandas as pd
from datetime import datetime

# Load the dataset
df = pd.read_csv(r'C:\Users\PURNANGSHU ROY\OneDrive\Desktop\car\cardata.csv')

# Convert 'year' to represent the age of the car
current_year = datetime.now().year
df['age_of_the_car'] = current_year - df['Year']

# Drop the 'year' feature
df.drop(columns=['Year'], inplace=True)

# Display the first few rows of the DataFrame after extracting the new feature and dropping 'year'
print(df.head())



# ### 7. Encode the categorical columns

# In[22]:


import pandas as pd

# Load the dataset
df = pd.read_csv(r'C:\Users\PURNANGSHU ROY\OneDrive\Desktop\car\cardata.csv')

# One-Hot Encoding for categorical columns
df = pd.get_dummies(df, columns=['Fuel_Type'])

# Label Encoding for ordinal categorical columns (if applicable)
# Example: 
# from sklearn.preprocessing import LabelEncoder
# label_encoder = LabelEncoder()
# df['ordinal_categorical_column'] = label_encoder.fit_transform(df['ordinal_categorical_column'])

# Display the first few rows of the DataFrame after encoding
print(df.head())


# ### 8. Separate the target and independent features.

# In[29]:


import pandas as pd

# Load the dataset
df = pd.read_csv(r'C:\Users\PURNANGSHU ROY\OneDrive\Desktop\car\cardata.csv')

# Print the column names to identify the correct target variable column
print(df.columns)

# Replace 'target_column' with the actual name of your target variable column
target_column = 'Selling_Price'

# Separate the target variable from the independent features
X = df.drop(columns=[target_column])  # Independent features
y = df[target_column]  # Target variable


# In[28]:


import pandas as pd

# Load the dataset
df = pd.read_csv(r'C:\Users\PURNANGSHU ROY\OneDrive\Desktop\car\cardata.csv')

# Assuming 'target_column' is the name of your target variable
target_column = 'Selling_Price'

# Separate the target variable from the independent features
X = df.drop(columns=[target_column])  # Independent features
y = df[target_column]  # Target variable

# Display the first few rows of the independent features and target variable
print("Independent Features (X):")
print(X.head())
print("\nTarget Variable (y):")
print(y.head())


# ### 9. Split the data into train and test.

# In[31]:


from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# ### 10. Build a Random forest Regressor model and check the r2-score for train and test.

# In[33]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# Load the dataset
df = pd.read_csv(r'C:\Users\PURNANGSHU ROY\OneDrive\Desktop\car\cardata.csv')

# Perform one-hot encoding for categorical columns
df = pd.get_dummies(df)

# Separate the target variable from the independent features
X = df.drop(columns=['Selling_Price'])  # Independent features
y = df['Selling_Price']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the Random Forest Regressor model
rf_regressor = RandomForestRegressor(random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)

# Predict on training and testing sets
y_train_pred = rf_regressor.predict(X_train)
y_test_pred = rf_regressor.predict(X_test)

# Calculate R^2 score for training and testing sets
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Print R^2 scores
print("R^2 score on training set:", r2_train)
print("R^2 score on testing set:", r2_test)


# ### 11. Create a pickle file with an extension as .pkl

# In[35]:


import pickle

# Assuming rf_regressor is your trained Random Forest Regressor model
# Instantiate the Random Forest Regressor model
rf_regressor = RandomForestRegressor(random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)

# Save the trained model to a file
with open('random_forest_regressor.pkl', 'wb') as file:
    pickle.dump(rf_regressor, file)


# ### 12. Create new folder/new project in visual studio/pycharm that should contain the "model.pkl" file *make sure you are using a virutal environment and install required packages.*

# ### a) Create a basic HTML form for the frontend

# Create a file **index.html** in the templates folder and copy the following code.

# In[52]:


<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Random Forest Regression Prediction</title>
</head>
<body>
    <h2>Random Forest Regression Prediction</h2>
    <form action="/predict" method="post">
        <label for="feature1">Feature 1:</label>
        <input type="text" id="feature1" name="feature1" required><br><br>

        <label for="feature2">Feature 2:</label>
        <input type="text" id="feature2" name="feature2" required><br><br>

        <!-- Add more input fields for additional features as needed -->

        <button type="submit">Predict</button>
    </form>
</body>
</html>



# ### b) Create app.py file and write the predict function

# In[48]:


pip install Flask


# In[51]:


from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model from the pickle file
with open('car.data', 'rb') as file:
    model = pickle.load(file)

# Create a Flask app
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for the predict function
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    features = [float(x) for x in request.form.values()]
    
    # Convert the input values to a numpy array
    input_features = np.array(features).reshape(1, -1)
    
    # Use the trained model to make a prediction
    prediction = model.predict(input_features)
    
    # Return the prediction as a response
    return 'Predicted Selling Price: {}'.format(prediction[0])

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)


# ### 13. Run the app.py python file which will render to index html page then enter the input values and get the prediction.

# In[ ]:





# ### Happy Learning :)

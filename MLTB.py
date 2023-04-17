from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB

# Define the Crypto Symbol
crypto_symbol = 'ETHUSDT'

# Import Bitcoin Price Data from binance
def get_binance_data(crypto_symbol):
    print(f"Getting {crypto_symbol} data from Binance...")
    # Construct the API URL
    url = f"https://api.binance.com/api/v3/klines?symbol={crypto_symbol}&interval=1d&limit=1000"
    response = requests.get(url)
    data = response.json()
    # if data is not empty
    if data:
        # Create a DataFrame
        df = pd.DataFrame(data)
        # Set the columns
        df.columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                      'Close Time', 'Quote Asset Volume', 'Number of Trades',
                      'Taker buy base asset volume', 'Taker buy quote asset volume',
                      'Ignore']
        # Convert the date columns to datetime
        df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
        df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')
        # Set the index to the date column
        df.set_index('Open Time', inplace=True)
        # Drop the ignore column
        df.drop('Ignore', axis=1, inplace=True)
        # Convert the columns to numeric
        df = df.apply(pd.to_numeric)
        # Return the DataFrame
        return df
    else:
        print('No data found')
        return None

# Save the data to a CSV file
def save_data(df, crypto_symbol):
    # Save the data to a CSV file
    df.to_csv(f'{crypto_symbol}.csv')
    print(f"Data saved to {crypto_symbol}.csv")

# Load the data from a CSV file
def load_data(crypto_symbol):
    # Load the data from a CSV file
    df = pd.read_csv(f'{crypto_symbol}.csv', index_col='Open Time', parse_dates=True)
    # Return the DataFrame
    return df

# Clean the data
def clean_data(df):
    # Drop the rows with missing values
    df.dropna(inplace=True)
    # Drop the duplicate rows
    df.drop_duplicates(inplace=True)
    # Return the cleaned DataFrame
    return df

# Train the model with the data
def train_model(df):
    # Create the features and target
    X = df.drop('Close', axis=1)
    y = df['Close']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Scale the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    # Create a list of models
    models = [('KNN', KNeighborsRegressor(n_neighbors=3)),
          ('Random Forest', RandomForestRegressor()),
          ('Decision Tree', DecisionTreeRegressor()),
          ('Linear Regression', LinearRegression()),
          ('Ridge', Ridge())]
    # Create a list to store the model scores
    model_scores = []
    # Loop through the models
    for name, model in models:
        # Fit the model to the data
        model.fit(X_train, y_train)
        # Evaluate the model and append its score to the list
        model_scores.append(model.score(X_test, y_test))
    # Return the model scores
    return model_scores


# Print the model scores
def print_model_scores(model_scores):
    # Create a list of model names
    model_names = ['KNN', 'SVR', 'Random Forest', 'Decision Tree', 'Linear Regression', 'Ridge']
    # Loop through the model names and scores and print them
    for name, score in zip(model_names, model_scores):
        print(f'{name} Model Accuracy: {score}')


# Get the data
df = get_binance_data(crypto_symbol)

# Save the data
save_data(df, crypto_symbol)

# Load the data
df = load_data(crypto_symbol)

# Clean the data
df = clean_data(df)

# Train the model and print the scores
model_scores = train_model(df)
print_model_scores(model_scores)

# Predict the price for the Next 30 Days using all models
def predict_price(df):
    # Create the features and target
    X = df.drop('Close', axis=1)
    y = df['Close']
    # Scale the data
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    # Create a list of models
    models = [('KNN', KNeighborsRegressor(n_neighbors=3)),
              ('Random Forest', RandomForestRegressor()),
              ('Decision Tree', DecisionTreeRegressor()),
              ('Linear Regression', LinearRegression()),
              ('Ridge', Ridge())]
    # Create a list to store the predictions
    predictions = []
    # Loop through the models
    for name, model in models:
        # Fit the model to the data
        model.fit(X, y)
        # Predict the price for the next 30 days
        prediction = model.predict(X[-30:])
        # Reshape prediction to be a row of a 2D array
        prediction = prediction.reshape(1, -1)
        # Append the prediction to the list
        predictions.append(prediction)
    # Concatenate the predictions into a single array
    predictions = np.concatenate(predictions, axis=1)
    # Return the predictions
    return predictions

predict_price(df)

# Save the predictions to a symbol name.csv file
def save_predictions(predictions, crypto_symbol):
    # Create a DataFrame
    df = pd.DataFrame(predictions)
    # Set the columns
    df.columns = ['KNN', 'Random Forest', 'Decision Tree', 'Linear Regression', 'Ridge']
    # Save the predictions to a CSV file
    df.to_csv(f'{crypto_symbol}_predictions.csv')
    print(f"Predictions saved to {crypto_symbol}_predictions.csv")

# Load the predictions from a CSV file
def load_predictions(crypto_symbol):
    # Load the predictions from a CSV file
    df = pd.read_csv(f'{crypto_symbol}_predictions.csv', index_col='Date')
    # Return the DataFrame
    return df

# Call the save_predictions function
save_predictions(predictions, crypto_symbol)
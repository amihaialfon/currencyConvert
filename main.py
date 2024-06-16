import requests
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta


def fetch_currency_data(api_key, base_currency, currencies):
    """
    Fetches the latest currency rates from FreeCurrencyAPI.
    """
    url = f"https://api.freecurrencyapi.com/v1/latest?apikey={api_key}&base_currency={base_currency}&currencies={','.join(currencies)}"
    response = requests.get(url)
    data = response.json()
    return data['data']


def predict_tomorrow_price(currency_data):
    """
    Predicts whether the currency will rise or fall tomorrow based on linear regression.
    """
    currencies = list(currency_data.keys())
    prices = list(currency_data.values())

    X = np.arange(len(prices)).reshape(-1, 1)
    y = np.array(prices)

    model = LinearRegression()
    model.fit(X, y)

    tomorrow = len(prices)
    tomorrow_price = model.predict([[tomorrow]])

    if tomorrow_price > prices[-1]:
        trend = "↑"
    elif tomorrow_price < prices[-1]:
        trend = "↓"
    else:
        trend = "↔"

    print("Linear Regression Model:")
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    print(f"Tomorrow's Price Prediction: {tomorrow_price[0]}")
    print(f"Trend for Tomorrow: {trend}")

    return tomorrow_price[0], trend


def main():
    api_key = ""
    base_currency = "ILS"
    currencies = ["EUR", "USD"]

    print("Fetching data from FreeCurrencyAPI...")
    currency_data = fetch_currency_data(api_key, base_currency, currencies)
    print("Fetched data:")
    for currency, price in currency_data.items():
        tonis = 1/price
        print(f"{currency}: now {price}, 1 {currency} is {tonis}₪")

    print("")
    print("Predicting tomorrow's price and trend...")
    tomorrow_price, trend = predict_tomorrow_price(currency_data)
    for currency in currency_data.keys():
        print(f"{currency}: expected to {trend} tomorrow")


if __name__ == "__main__":
    main()

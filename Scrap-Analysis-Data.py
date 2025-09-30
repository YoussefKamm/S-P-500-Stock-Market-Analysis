import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf

# Function to scrape S&P 500 symbols
def scrape_sp500_symbols(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    symbols = []
    table = soup.find('table', {'class': 'wikitable'})

    if table:
        rows = table.find_all('tr')[1:]
        for row in rows:
            cols = row.find_all('td')
            if len(cols) > 0:
                symbol = cols[0].text.strip()
                name = cols[1].text.strip()
                symbols.append((symbol, name))

    symbols_df = pd.DataFrame(symbols, columns=['Symbol', 'Name'])
    symbols_df['ID'] = range(1, len(symbols_df) + 1)
    symbols_df = symbols_df[['ID', 'Symbol', 'Name']]

    return symbols_df

# Scraping S&P 500 symbols
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
symbols_df = scrape_sp500_symbols(url)

# Function to get historical prices for all symbols
def get_historical_prices(symbols_df, start_date, end_date):
    historical_data = []

    for index, row in symbols_df.iterrows():
        symbol = row['Symbol']
        try:
            # Fetch historical prices
            stock_data = yf.download(symbol, start=start_date, end=end_date)
            stock_data.reset_index(inplace=True)

            for _, price_row in stock_data.iterrows():
                historical_data.append({
                    'ID': row['ID'],
                    'Symbol': symbol,
                    'Date': price_row['Date'],
                    'Open': price_row['Open'],
                    'High': price_row['High'],
                    'Low': price_row['Low'],
                    'Close': price_row['Close'],
                    'Volume': price_row['Volume']
                })
        except Exception as e:
            print(f"Error retrieving data for {symbol}: {e}")

    return pd.DataFrame(historical_data)

# Define the date range
start_date = '2018-01-01'
end_date = '2024-09-30'

# Get historical prices DataFrame
historical_prices_df = get_historical_prices(symbols_df, start_date, end_date)

# Function to calculate metrics with moving averages
def calculate_metrics(historical_prices_df):
    metrics_data = []

    # Calculate moving averages (1 to 50 days) for each stock symbol
    for (stock_id, symbol), group in historical_prices_df.groupby(['ID', 'Symbol']):
        for i in range(1, 51):
            group[f'Moving-Avg-{i}'] = group['Close'].rolling(window=i).mean()

        # Store the required columns for calculated_metrics_df
        for _, row in group.iterrows():
            metrics_data.append({
                'ID': stock_id,
                'Symbol': symbol,
                'Date': row['Date'],
                'Close': row['Close'],
                **{f'Moving-Avg-{i}': row[f'Moving-Avg-{i}'] for i in range(1, 51)}
            })

    return pd.DataFrame(metrics_data)

# Create calculated_metrics_df
calculated_metrics_df = calculate_metrics(historical_prices_df)



# Save DataFrames to CSV
symbols_df.to_csv('sp500_symbols.csv', index=False)
historical_prices_df.to_csv('historical_prices.csv', index=False)
calculated_metrics_df.to_csv('calculated_metrics.csv', index=False)


print("\n 3 CSV files created successfully!")
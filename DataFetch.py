import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import concurrent.futures

def acquire_and_process_stock_data(ticker, years=10):
    print(f"Processing: {ticker}...")    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    
    if stock_data.empty:
        print(f"-----> No data found for {ticker}, skipping.")
        return None
        
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)
    
    stock_data['Ticker'] = ticker
    stock_data.reset_index(inplace=True)

    stock_data['Daily_Return'] = stock_data['Close'].pct_change()
    stock_data['Daily_Range'] = stock_data['High'] - stock_data['Low']
    stock_data['Price_Change'] = stock_data['Close'] - stock_data['Open']
    stock_data['Day_of_Week'] = pd.to_datetime(stock_data['Date']).dt.dayofweek
    stock_data['Month'] = pd.to_datetime(stock_data['Date']).dt.month
    stock_data['Year'] = pd.to_datetime(stock_data['Date']).dt.year

    
    stock_data['Daily_Volatility'] = stock_data['Daily_Return'].rolling(window=20).std()
    
    
    stock_data['SMA_20'] = stock_data.rolling(window=20)['Close'].mean()
    stock_data['SMA_50'] = stock_data.rolling(window=50)['Close'].mean()
    stock_data['SMA_200'] = stock_data.rolling(window=200)['Close'].mean()
    
    
    stock_data['EMA_12'] = stock_data['Close'].ewm(span=12, adjust=False).mean()
    stock_data['EMA_26'] = stock_data['Close'].ewm(span=26, adjust=False).mean()
    
    stock_data['MACD'] = stock_data['EMA_12'] - stock_data['EMA_26']
    stock_data['MACD_Signal'] = stock_data['MACD'].ewm(span=9, adjust=False).mean()
    
    delta = stock_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    stock_data['RSI'] = 100 - (100 / (1 + rs))
    
    bb_std = stock_data.rolling(window=20)['Close'].std()
    stock_data['BB_Upper'] = stock_data['SMA_20'] + (bb_std * 2)
    stock_data['BB_Lower'] = stock_data['SMA_20'] - (bb_std * 2)
    stock_data['BB_Width'] = stock_data['BB_Upper'] - stock_data['BB_Lower']
    stock_data['OBV'] = (stock_data['Volume'] * (~stock_data['Close'].diff().le(0) * 2 - 1)).cumsum()
    stock_data['Target'] = (stock_data['Close'].shift(-1) > stock_data['Close']).astype(int)
    stock_data.dropna(inplace=True)
    
    return stock_data


def fetch_all_stocks_concurrently(tickers, years=10, max_workers=10):
    print(f"\nStarting concurrent data fetch for {len(tickers)} tickers with {max_workers} workers...")
    processed_data_list = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        
        future_to_ticker = {executor.submit(acquire_and_process_stock_data, ticker, years): ticker for ticker in tickers}
        
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            try:
                result_df = future.result()
                if result_df is not None:
                    processed_data_list.append(result_df)
            except Exception as exc:
                ticker = future_to_ticker[future]
                print(f'{ticker} generated an exception: {exc}')
                
    if not processed_data_list:
        print("No data was successfully processed.")
        return None

    print("\nCombining all data into a single DataFrame...")
    master_df = pd.concat(processed_data_list, ignore_index=True) 
    master_df.sort_values(by=['Ticker', 'Date'], inplace=True)
    
    return master_df


if __name__ == "__main__":
    
    tickers = [
        
        '^GSPC', '^DJI', '^IXIC', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'CRM', 'AMD', 'JPM', 'V', 'MA', 'BAC', 'GS', 'UNH', 'JNJ', 'LLY', 'PFE', 'MRK', 'WMT', 'PG', 'COST', 'HD', 'NKE'
    ]
    
    
    master_dataframe = fetch_all_stocks_concurrently(tickers, years=10, max_workers=10)
    
    if master_dataframe is not None:
        
        output_filename = "combined_stock_data_with_features.csv"
        master_dataframe.to_csv(output_filename, index=False)
        
        print(f"\n{'='*60}")
        print("Data acquisition and processing complete!")
        print(f"Successfully processed data for {master_dataframe['Ticker'].nunique()} out of {len(tickers)} tickers.")
        print(f"Total records in the final dataset: {len(master_dataframe)}")
        print(f"Data saved to '{output_filename}'")
        print(f"{'='*60}\n")

        print("Sample of the final DataFrame (first 5 rows):")
        print(master_dataframe.head())
        print("\nSample of the final DataFrame (last 5 rows):")
        print(master_dataframe.tail())


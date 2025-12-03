import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import concurrent.futures

def acquire_and_process_stock_data(ticker, years=10):
    print(f"Processing: {ticker}...")
        
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    try:    
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if stock_data.empty:
            print(f"-----> No data found for {ticker}, skipping.")
            return None

        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.get_level_values(0)
        
        if stock_data.index.tz is not None:
            stock_data.index = stock_data.index.tz_localize(None)

        ticker_obj = yf.Ticker(ticker)
        actions = ticker_obj.actions  
        
        if not actions.empty:
            
            if actions.index.tz is not None:
                actions.index = actions.index.tz_localize(None)

            stock_data = stock_data.join(actions, how='left')
            
            if 'Dividends' in stock_data.columns:
                stock_data['Dividends'] = stock_data['Dividends'].fillna(0.0)
            else:
                stock_data['Dividends'] = 0.0
                
            if 'Stock Splits' in stock_data.columns:
                stock_data['Stock Splits'] = stock_data['Stock Splits'].fillna(0.0)
            else:
                stock_data['Stock Splits'] = 0.0
        else:
            stock_data['Dividends'] = 0.0
            stock_data['Stock Splits'] = 0.0

        
        stock_data['Ticker'] = ticker
        stock_data.reset_index(inplace=True) 
        close_col = stock_data['Close']
        
        if isinstance(close_col, pd.DataFrame):
            close_col = close_col.iloc[:, 0]
            
            stock_data = stock_data.loc[:, ~stock_data.columns.duplicated()]
            stock_data['Close'] = close_col

        stock_data['Tomorrow'] = close_col.shift(-1)
        stock_data['Target'] = (stock_data['Tomorrow'] > stock_data['Close']).astype(int)
        
        
        required_cols = [
            'Date', 'Ticker', 
            'Open', 'High', 'Low', 'Close', 'Volume', 
            'Dividends', 'Stock Splits', 
            'Tomorrow', 'Target'
        ]
        
        
        available_cols = [c for c in required_cols if c in stock_data.columns]
        stock_data = stock_data[available_cols]

        stock_data.dropna(inplace=True)
        
        return stock_data

    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None


def fetch_all_stocks_concurrently(tickers, years=10, max_workers=10):
    print(f"\nStarting concurrent data fetch for {len(tickers)} tickers...")
    processed_data_list = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(acquire_and_process_stock_data, ticker, years): ticker for ticker in tickers}
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            try:
                result_df = future.result()
                if result_df is not None:
                    processed_data_list.append(result_df)
            except Exception as exc:
                print(f'Exception generated: {exc}')
                
    if not processed_data_list:
        print("No data was successfully processed.")
        return None

    print("\nCombining data...")
    master_df = pd.concat(processed_data_list, ignore_index=True)
    
    
    if 'Date' in master_df.columns and 'Ticker' in master_df.columns:
        master_df.sort_values(by=['Ticker', 'Date'], inplace=True)
    
    return master_df


if __name__ == "__main__":
    
    tickers = [
        '^GSPC', '^DJI', '^IXIC', 
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'CRM', 'AMD',
        'JPM', 'V', 'MA', 'BAC', 'GS',
        'UNH', 'JNJ', 'LLY', 'PFE', 'MRK',
        'WMT', 'PG', 'COST', 'HD', 'NKE'
    ]
    
    master_dataframe = fetch_all_stocks_concurrently(tickers, years=10)
    
    if master_dataframe is not None:
        output_filename = "combined_stock_data.csv"
        master_dataframe.to_csv(output_filename, index=False)
        
        print(f"\n{'='*60}")
        print(f"Data saved to '{output_filename}'")
        print(f"Columns: {list(master_dataframe.columns)}")
        print(f"Total rows: {len(master_dataframe)}")
        print(f"{'='*60}\n")
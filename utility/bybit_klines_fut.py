import pandas as pd
import numpy as np
import time
import os
from tqdm import tqdm 
import pytz
from datetime import datetime, timedelta
import logging

# Configurazione logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bybit_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def convert_interval(interval):
    """
    Converti il formato dell'intervallo da stringa (es. "1h") a valore numerico per Bybit API
    """
    mapping = {
        "1m": "1",
        "3m": "3",
        "5m": "5", 
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "2h": "120",
        "4h": "240",
        "6h": "360", 
        "12h": "720",
        "1d": "D",
        "1w": "W",
        "1M": "M"
    }
    return mapping.get(interval, interval)

def download_single_symbol_kline(session, symbol, interval="1h", start_date=None, end_date=None, output_dir="kline_data"):
    """
    Download kline data for a single symbol
    
    Args:
        session: pybit HTTP session
        symbol: trading pair symbol (e.g., "BTCUSDT")
        interval: kline interval ("1h", "4h", "1d", etc.)
        start_date: start date (YYYY-MM-DD)
        end_date: end date (YYYY-MM-DD)
        output_dir: directory to save the data
        
    Returns:
        pd.DataFrame: DataFrame with kline data or None if no data
    """
    logger.info(f"Downloading kline data for {symbol} (interval: {interval})")
    
    # Converti il formato dell'intervallo per Bybit API
    bybit_interval = convert_interval(interval)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file path
    output_path = f"{output_dir}/{symbol}_{interval}_kline.csv"
    
    # Check if file already exists
    if os.path.exists(output_path):
        try:
            df_existing = pd.read_csv(output_path)
            if not df_existing.empty:
                logger.info(f"Loading existing data for {symbol} from {output_path}: {len(df_existing)} records")
                if 'timestamp' in df_existing.columns:
                    df_existing['timestamp'] = pd.to_datetime(df_existing['timestamp'])
                return df_existing
        except Exception as e:
            logger.warning(f"Could not load existing file for {symbol}: {e}")
    
    # Convert dates to timestamps
    if start_date:
        start_dt = pd.to_datetime(start_date)
    else:
        # Default: last 30 days
        start_dt = datetime.now() - timedelta(days=30)
    
    if end_date:
        end_dt = pd.to_datetime(end_date)
    else:
        # Default: current date
        end_dt = datetime.now()
    
    # Add timezone if not present
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=pytz.UTC)
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=pytz.UTC)
    
    # Convert to millisecond timestamps
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    
    logger.info(f"Date range: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
    
    # Maximum span per request (to avoid hitting limits)
    max_span_days = 7
    
    # Calculate step size in milliseconds
    step_ms = max_span_days * 24 * 60 * 60 * 1000
    
    # Initialize an empty list to store all kline data
    all_data = []
    
    # Split the date range into smaller chunks
    current_start = start_ms
    
    while current_start < end_ms:
        # Calculate current end (not exceeding the final end date)
        current_end = min(current_start + step_ms, end_ms)
        
        current_start_dt = datetime.fromtimestamp(current_start / 1000).strftime('%Y-%m-%d')
        current_end_dt = datetime.fromtimestamp(current_end / 1000).strftime('%Y-%m-%d')
        
        logger.info(f"Fetching chunk for {symbol}: {current_start_dt} to {current_end_dt}")
        
        # Add retries for robustness
        max_retries = 3
        success = False
        
        for retry in range(max_retries):
            try:
                # Make the API request
                result = session.get_kline(
                    category="linear",
                    symbol=symbol,
                    interval=bybit_interval,
                    start=current_start,
                    end=current_end,
                    limit=1000
                )
                
                if result.get("retCode") == 0 and result.get("result", {}).get("list"):
                    chunk_data = result["result"]["list"]
                    # logger.info(f"Got {len(chunk_data)} records for {symbol}")
                    all_data.extend(chunk_data)
                    success = True
                    break
                else:
                    logger.warning(f" Attempt {retry+1}/{max_retries}: Error or no data for {symbol}: {result}")
                    
            except Exception as e:
                logger.warning(f"Attempt {retry+1}/{max_retries}: Exception for {symbol}: {e}")
            
            # Wait before retry
            time.sleep(0.2)
        
        if not success:
            logger.warning(f"Failed to fetch data for {symbol} chunk {current_start_dt} to {current_end_dt} after {max_retries} attempts")
        
        # Move to next chunk
        current_start = current_end + 1
        
        # Respect rate limits
        time.sleep(0.2)
    
    # Convert to DataFrame if we have data
    if all_data:
        logger.info(f"Total records collected for {symbol}: {len(all_data)}")
        
        # Determine columns based on the actual data
        if len(all_data[0]) >= 6:
            # Define expected columns
            expected_columns = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
            # Use only as many columns as we have in the data
            columns = expected_columns[:len(all_data[0])]
            
            # Create DataFrame
            df = pd.DataFrame(all_data, columns=columns)
            
            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
            
            # Convert numeric columns
            for col in df.columns:
                if col != "timestamp":
                    df[col] = pd.to_numeric(df[col])
            
            # Sort by timestamp
            df = df.sort_values("timestamp")
            
            # Add symbol column
            df["symbol"] = symbol
            
            # Save to CSV
            df.to_csv(output_path, index=False)
            logger.info(f"Data saved to {output_path}: {len(df)} records")
            
            return df
        else:
            logger.warning(f"Unknown data format for {symbol}, cannot create DataFrame")
    else:
        logger.warning(f"No data collected for {symbol}")
    
    return None

def fetch_all_kline_data(session, symbols_df, interval="1h", start_date=None, end_date=None, output_dir="kline_data", max_symbols=None):
    """
    Fetch kline (candlestick) data for all symbols within the specified period
    
    Args:
        session: HTTP session for Bybit API
        symbols_df (pd.DataFrame): DataFrame with symbols
        interval (str): Kline interval. Options: '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w', '1M'
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        output_dir (str): Directory to save CSV files
        max_symbols (int, optional): Maximum number of symbols to process
        
    Returns:
        dict: Dictionary with symbols as keys and kline DataFrames as values
    """
    logger.info(f"Starting fetch_all_kline_data: interval={interval}, start={start_date}, end={end_date}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a summary file
    summary_file = f"{output_dir}/kline_data_summary_{interval}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"KLINE DATA SUMMARY - INTERVAL: {interval}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date Range: {start_date} to {end_date}\n")
        f.write(f"Interval: {interval}\n\n")
        f.write("Symbol\tStart Date\tEnd Date\tCandles\tStart Price\tEnd Price\tReturn\n")
        f.write("-" * 100 + "\n")
    
    # First test with BTCUSDT to make sure the API is working
    logger.info("Testing API with BTCUSDT...")
    btc_test = download_single_symbol_kline(
        session=session,
        symbol="BTCUSDT",
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir
    )
    
    if btc_test is None or btc_test.empty:
        logger.warning("Test with BTCUSDT failed or returned no data. Will proceed anyway with all symbols.")
    else:
        logger.info(f"Test with BTCUSDT successful: {len(btc_test)} records")
    
    # Dictionary to store kline data
    kline_data = {}
    if btc_test is not None and not btc_test.empty:
        kline_data["BTCUSDT"] = btc_test
    
    # Limit the number of symbols if specified
    if max_symbols is not None and max_symbols > 0:
        if len(symbols_df) > max_symbols:
            symbols_df = symbols_df.head(max_symbols)
            logger.info(f"Limited to first {max_symbols} symbols")
    
    logger.info(f"Processing {len(symbols_df)} symbols")
    
    # Iterate over all symbols with a progress bar (skip BTCUSDT if already processed)
    symbols_to_process = [row['symbol'] for _, row in symbols_df.iterrows() if row['symbol'] != "BTCUSDT" or "BTCUSDT" not in kline_data]
    
    for symbol in tqdm(symbols_to_process, desc=f"Fetching {interval} kline data"):
        logger.info(f"Processing symbol: {symbol}")
        
        # Download kline data for this symbol
        df = download_single_symbol_kline(
            session=session,
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir
        )
        
        # Add to our collection if we got data
        if df is not None and not df.empty:
            kline_data[symbol] = df
            
            # Update the summary file
            with open(summary_file, 'a') as f:
                start_price = df['open'].iloc[0] if not df.empty else "N/A"
                end_price = df['close'].iloc[-1] if not df.empty else "N/A"
                
                # Calculate return
                if isinstance(start_price, (int, float)) and isinstance(end_price, (int, float)) and start_price > 0:
                    ret = ((end_price / start_price) - 1) * 100
                    ret_str = f"{ret:.2f}%"
                else:
                    ret_str = "N/A"
                
                f.write(f"{symbol}\t{df['timestamp'].min() if not df.empty else 'N/A'}\t" +
                        f"{df['timestamp'].max() if not df.empty else 'N/A'}\t" +
                        f"{len(df)}\t{start_price}\t{end_price}\t{ret_str}\n")
    
    # Add overall summary information
    with open(summary_file, 'a') as f:
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"Total Symbols Processed: {len(symbols_df)}\n")
        f.write(f"Total Symbols with Data: {len(kline_data)}\n")
        f.write(f"Total Records: {sum(len(df) for df in kline_data.values())}\n")
        
        # Calculate date ranges across all symbols
        if kline_data:
            all_timestamps = []
            for df in kline_data.values():
                all_timestamps.extend(df['timestamp'].tolist())
            
            if all_timestamps:
                f.write(f"Earliest Data Point: {min(all_timestamps)}\n")
                f.write(f"Latest Data Point: {max(all_timestamps)}\n")
                
                # List symbols with most data
                symbols_by_records = {symbol: len(df) for symbol, df in kline_data.items()}
                top_symbols = sorted(symbols_by_records.items(), key=lambda x: x[1], reverse=True)[:10]
                
                f.write("\nTop 10 Symbols by Number of Records:\n")
                for symbol, count in top_symbols:
                    f.write(f"{symbol}: {count} candles\n")
    
    logger.info(f"Completed fetch_all_kline_data: {len(kline_data)} symbols with data")
    logger.info(f"Kline data summary saved to {summary_file}")
    return kline_data
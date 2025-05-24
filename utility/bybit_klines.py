import pandas as pd
import numpy as np
import time
import os
from tqdm import tqdm 
import pytz
from datetime import datetime, timedelta
import logging
import threading
import queue
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bybit_parallel.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global rate limiter for API requests
class RateLimiter:
    def __init__(self, max_calls_per_second=5):
        """
        Initialize a rate limiter to prevent hitting API limits
        
        Args:
            max_calls_per_second (int): Maximum API calls allowed per second
        """
        self.max_calls_per_second = max_calls_per_second
        self.calls_timestamps = []
        self.lock = threading.Lock()
        
    def acquire(self):
        """Wait until a call is allowed under the rate limit"""
        with self.lock:
            now = time.time()
            
            # Remove timestamps older than 1 second
            self.calls_timestamps = [ts for ts in self.calls_timestamps if now - ts < 1]
            
            # If at the limit, wait
            if len(self.calls_timestamps) >= self.max_calls_per_second:
                sleep_time = 1 - (now - self.calls_timestamps[0])
                if sleep_time > 0:
                    time.sleep(sleep_time + random.uniform(0.01, 0.05))  # Add small random delay for better distribution
            
            # Record the call
            self.calls_timestamps.append(time.time())

# Global rate limiter instance
rate_limiter = RateLimiter(max_calls_per_second=3)  # Adjust this value based on Bybit's limits

def convert_interval(interval):
    """
    Convert interval format from string (e.g., "1h") to numeric value for Bybit API
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
def download_kline_data(session, symbol, interval="1h", start_date=None, end_date=None, output_dir="kline_data", category="linear", launch_time=None):
    """
    Download kline data for a single symbol
    
    Args:
        session: pybit HTTP session
        symbol: trading pair symbol (e.g., "BTCUSDT")
        interval: kline interval ("1h", "4h", "1d", etc.)
        start_date: start date (YYYY-MM-DD)
        end_date: end date (YYYY-MM-DD)
        output_dir: directory to save the data
        category: "linear" for futures, "spot" for spot
        launch_time: launch time of the symbol (datetime)
        
    Returns:
        pd.DataFrame: DataFrame with kline data or None if no data
    """
    logger.info(f"Downloading {category} kline data for {symbol} (interval: {interval})")
    
    # Convert interval format for Bybit API
    bybit_interval = convert_interval(interval)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file path
    output_path = f"{output_dir}/{symbol}_{interval}_{category}_kline.csv"
    
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
    
    # Adjust start date based on launch time if provided
    if launch_time:
        # Make sure launch_time is a datetime
        if isinstance(launch_time, str):
            launch_time = pd.to_datetime(launch_time)
            
        # Add timezone if not present
        if launch_time.tzinfo is None:
            launch_time = launch_time.replace(tzinfo=pytz.UTC)
            
        # If launch time is after start_dt, use launch time instead
        if launch_time > start_dt:
            start_dt = launch_time
            logger.info(f"Adjusted start date to launch time for {symbol}: {start_dt}")
            
        # If launch time is after end_dt, skip this symbol
        if launch_time > end_dt:
            logger.info(f"Skipping {symbol} as launch time ({launch_time}) is after end date ({end_dt})")
            return None
    
    # Determine the actual required start date
    required_start_dt = start_dt
    
    # Variable to hold existing data if partial
    df_existing = None
    need_to_download_missing = False
    download_end_dt = end_dt  # By default download until end_date
    
    # Thread-safe file check to avoid race conditions
    file_lock = threading.Lock()
    with file_lock:
        # Check if file already exists
        if os.path.exists(output_path):
            try:
                df_existing_temp = pd.read_csv(output_path)
                if not df_existing_temp.empty:
                    if 'timestamp' in df_existing_temp.columns:
                        df_existing_temp['timestamp'] = pd.to_datetime(df_existing_temp['timestamp'])
                        # Ensure timezone
                        if df_existing_temp['timestamp'].dt.tz is None:
                            df_existing_temp['timestamp'] = df_existing_temp['timestamp'].dt.tz_localize(pytz.UTC)
                        
                        # Check if existing data covers the required period
                        existing_min_date = df_existing_temp['timestamp'].min()
                        
                        if existing_min_date <= required_start_dt:
                            # Existing data covers the required period
                            logger.info(f"Loading existing data for {symbol} from {output_path}: {len(df_existing_temp)} records (covers required period)")
                            return df_existing_temp
                        else:
                            # Need to download missing data from the beginning
                            logger.info(f"Existing data for {symbol} starts from {existing_min_date}, but need data from {required_start_dt}")
                            df_existing = df_existing_temp
                            need_to_download_missing = True
                            # Download only up to the existing data start
                            download_end_dt = existing_min_date - timedelta(minutes=1)
                    else:
                        logger.info(f"Loading existing data for {symbol} from {output_path}: {len(df_existing_temp)} records")
                        return df_existing_temp
            except Exception as e:
                logger.warning(f"Could not load existing file for {symbol}: {e}")
    
    # Convert to millisecond timestamps
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(download_end_dt.timestamp() * 1000)
    
    logger.info(f"Date range for {symbol}: {start_dt.strftime('%Y-%m-%d')} to {download_end_dt.strftime('%Y-%m-%d')}")
    
    # Maximum span per request (to avoid hitting limits)
    max_span_days = 7
    
    # Calculate step size in milliseconds
    step_ms = max_span_days * 24 * 60 * 60 * 1000
    
    # Initialize an empty list to store all kline data
    all_data = []
    
    # Counter for consecutive empty responses
    consecutive_empty_chunks = 0
    max_empty_chunks = 3  # After 3 consecutive empty chunks, assume symbol doesn't exist
    
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
        chunk_has_data = False
        
        for retry in range(max_retries):
            try:
                # Apply rate limiting
                rate_limiter.acquire()
                
                # Make the API request
                result = session.get_kline(
                    category=category,
                    symbol=symbol,
                    interval=bybit_interval,
                    start=current_start,
                    end=current_end,
                    limit=1000
                )
                
                if result.get("retCode") == 0:
                    chunk_data = result.get("result", {}).get("list", [])
                    if chunk_data:
                        logger.info(f"  ✓ Got {len(chunk_data)} records for {symbol}")
                        all_data.extend(chunk_data)
                        success = True
                        chunk_has_data = True
                        consecutive_empty_chunks = 0  # Reset counter
                    else:
                        # Empty list but successful API call
                        logger.info(f"  ⚠️  No data for {symbol} in period {current_start_dt} to {current_end_dt}")
                        success = True  # API call was successful, just no data
                        chunk_has_data = False
                    break
                else:
                    logger.warning(f"  ✗ Attempt {retry+1}/{max_retries}: Error for {symbol}: {result}")
                    
            except Exception as e:
                error_msg = str(e)
                # Check if it's a "not supported symbol" error
                if "Not supported symbols" in error_msg or "ErrCode: 10001" in error_msg:
                    logger.warning(f"  ✗ Symbol {symbol} not supported in {category} market")
                    # Don't retry for unsupported symbols
                    return None
                else:
                    logger.warning(f"  ✗ Attempt {retry+1}/{max_retries}: Exception for {symbol}: {e}")
            
            # Wait before retry with backoff
            if retry < max_retries - 1:  # Don't wait after last retry
                time.sleep(1 * (2 ** retry))
        
        if not success:
            logger.warning(f"Failed to fetch data for {symbol} chunk {current_start_dt} to {current_end_dt} after {max_retries} attempts")
            # Consider this as an empty chunk
            consecutive_empty_chunks += 1
        elif not chunk_has_data:
            consecutive_empty_chunks += 1
        
        # Check if we've had too many consecutive empty chunks
        if consecutive_empty_chunks >= max_empty_chunks:
            logger.warning(f"  ✗ {consecutive_empty_chunks} consecutive empty chunks for {symbol}. Symbol likely doesn't exist in {category} market.")
            return None
        
        # Move to next chunk
        current_start = current_end + 1
    
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
            
            # If we had existing data that didn't cover the full period, merge it
            if need_to_download_missing and df_existing is not None and not df_existing.empty:
                # Combine new and existing data
                df = pd.concat([df, df_existing], ignore_index=True)
                # Remove duplicates, keeping the first occurrence
                df = df.drop_duplicates(subset=['timestamp'], keep='first')
                # Sort again
                df = df.sort_values("timestamp")
                logger.info(f"Merged new data with existing data for {symbol}: total {len(df)} records")
            
            # Thread-safe file writing
            with file_lock:
                # Save to CSV
                df.to_csv(output_path, index=False)
                logger.info(f"Data saved to {output_path}: {len(df)} records")
            
            return df
        else:
            logger.warning(f"Unknown data format for {symbol}, cannot create DataFrame")
    else:
        logger.warning(f"No data collected for {symbol}")
    
    return None

def generate_spot_symbol(future_symbol):
    """
    Generate spot symbol corresponding to a future symbol by removing numeric prefixes
    
    Examples:
    - 1000PEPEUSDT -> PEPEUSDT (removes '1000')
    - 1000000BABYDOGEUSDT -> BABYDOGEUSDT (removes '1000000')
    - BTCUSDT -> BTCUSDT (remains the same)
    """
    # Remove any PERP or PERPETUAL suffixes
    if future_symbol.endswith("PERP"):
        base_symbol = future_symbol[:-4]
    elif future_symbol.endswith("PERPETUAL"):
        base_symbol = future_symbol[:-9]
    else:
        base_symbol = future_symbol
    
    # Extract numeric prefix using regex
    import re
    match = re.match(r'^(\d+)([A-Za-z].+)$', base_symbol)
    if match:
        numeric_prefix = match.group(1)
        token_part = match.group(2)
        return token_part
    
    # If no numeric prefix found, return the original symbol
    return base_symbol

def map_future_to_spot(symbols_df):
    """
    Create mapping between future symbols and spot symbols
    
    Args:
        symbols_df (pd.DataFrame): DataFrame with future symbols
        
    Returns:
        pd.DataFrame: DataFrame with future symbols and corresponding spot symbols
    """
    # Create a copy of the DataFrame
    mapping_df = symbols_df.copy()
    
    # Add column for spot symbol
    mapping_df['spotSymbol'] = mapping_df['symbol'].apply(generate_spot_symbol)
    
    # Save the mapping for debugging
    mapping_path = "future_to_spot_mapping_debug.csv"
    mapping_df[['symbol', 'spotSymbol']].to_csv(mapping_path, index=False)
    input(mapping_df)
    
    return mapping_df

def parallel_download_data(session, symbols_info, interval="1h", start_date=None, end_date=None, 
                           output_dir="kline_data", category="linear", max_workers=5):
    """
    Download kline data for multiple symbols in parallel
    
    Args:
        session: pybit HTTP session
        symbols_info: list of tuples (symbol, launch_time) to download
        interval: kline interval ("1h", "4h", "1d", etc.)
        start_date: start date (YYYY-MM-DD)
        end_date: end date (YYYY-MM-DD)
        output_dir: directory to save the data
        category: "linear" for futures, "spot" for spot
        max_workers: maximum number of parallel workers
        
    Returns:
        dict: Dictionary with symbols as keys and kline DataFrames as values
    """
    # Dictionary to store results
    results = {}
    
    # Create a queue for symbols
    symbols_queue = queue.Queue()
    for symbol_info in symbols_info:
        symbols_queue.put(symbol_info)
    
    # Create a thread-safe dictionary for results
    results_lock = threading.Lock()
    
    def worker():
        """Worker function for thread pool"""
        while not symbols_queue.empty():
            try:
                symbol_info = symbols_queue.get(block=False)
                if isinstance(symbol_info, tuple) and len(symbol_info) == 2:
                    symbol, launch_time = symbol_info
                else:
                    symbol = symbol_info
                    launch_time = None
            except queue.Empty:
                break
                
            try:
                df = download_kline_data(
                    session=session,
                    symbol=symbol,
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date,
                    output_dir=output_dir,
                    category=category,
                    launch_time=launch_time
                )
                
                if df is not None and not df.empty:
                    with results_lock:
                        results[symbol] = df
                
            except Exception as e:
                logger.error(f"Error downloading {symbol}: {e}")
            finally:
                symbols_queue.task_done()
    
    # Create and start worker threads
    threads = []
    for _ in range(min(max_workers, len(symbols_info))):
        t = threading.Thread(target=worker)
        t.daemon = True
        t.start()
        threads.append(t)
    
    # Wait for all tasks to complete
    for t in threads:
        t.join()
    
    return results

def fetch_all_data_parallel(session, symbols_df, interval="1h", start_date=None, end_date=None, 
                           output_dir="kline_data", category="linear", max_workers=5, max_symbols=None):
    """
    Fetch kline data for all symbols in parallel
    
    Args:
        session: HTTP session for Bybit API
        symbols_df (pd.DataFrame): DataFrame with symbols
        interval (str): Kline interval. Options: '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w', '1M'
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        output_dir (str): Directory to save CSV files
        category (str): "linear" for futures, "spot" for spot
        max_workers (int): Maximum number of parallel workers
        max_symbols (int, optional): Maximum number of symbols to process
        
    Returns:
        dict: Dictionary with symbols as keys and kline DataFrames as values
    """
    logger.info(f"Starting parallel data download: interval={interval}, category={category}, max_workers={max_workers}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a summary file
    summary_file = f"{output_dir}/{category}_data_summary_{interval}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"{category.upper()} DATA SUMMARY - INTERVAL: {interval}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date Range: {start_date} to {end_date}\n")
        f.write(f"Interval: {interval}\n\n")
        f.write("Symbol\tStart Date\tEnd Date\tCandles\tStart Price\tEnd Price\tReturn\n")
        f.write("-" * 100 + "\n")
    
    # First test with BTCUSDT to make sure the API is working
    logger.info(f"Testing API with BTCUSDT {category}...")
    btc_test = download_kline_data(
        session=session,
        symbol="BTCUSDT",
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        category=category,
        launch_time=None  # BTCUSDT has been around for a long time
    )
    
    if btc_test is None or btc_test.empty:
        logger.warning(f"Test with BTCUSDT {category} failed or returned no data. Will proceed anyway with all symbols.")
    else:
        logger.info(f"Test with BTCUSDT {category} successful: {len(btc_test)} records")
    
    # Dictionary to store kline data
    kline_data = {}
    if btc_test is not None and not btc_test.empty:
        kline_data["BTCUSDT"] = btc_test
    
    # Process symbols_df based on category
    if category == "spot":
        # For spot, create mapping from future to spot symbols
        processed_df = map_future_to_spot(symbols_df)
        # Save the mapping for reference
        mapping_path = f"{output_dir}/future_to_spot_mapping.csv"
        processed_df[['symbol', 'spotSymbol']].to_csv(mapping_path, index=False)
        logger.info(f"Saved symbol mapping to {mapping_path}")
        
        # Create a list of tuples (spot_symbol, launch_time)
        symbols_to_download = []
        for _, row in processed_df.iterrows():
            spot_symbol = row['spotSymbol']
            # Use the same launch time for spot as for future
            launch_time = row.get('launchTime', None)
            symbols_to_download.append((spot_symbol, launch_time))
    else:
        # For futures, use symbol column directly
        processed_df = symbols_df
        # Create a list of tuples (symbol, launch_time)
        symbols_to_download = []
        for _, row in processed_df.iterrows():
            symbol = row['symbol']
            launch_time = row.get('launchTime', None)
            symbols_to_download.append((symbol, launch_time))
    
    # Remove BTCUSDT if already processed
    symbols_to_download = [s for s in symbols_to_download if s[0] != "BTCUSDT" or "BTCUSDT" not in kline_data]
    
    # Limit the number of symbols if specified
    if max_symbols is not None and max_symbols > 0:
        if len(symbols_to_download) > max_symbols - (1 if "BTCUSDT" in kline_data else 0):
            symbols_to_download = symbols_to_download[:max_symbols - (1 if "BTCUSDT" in kline_data else 0)]
            logger.info(f"Limited to {max_symbols} symbols including {'BTCUSDT ' if 'BTCUSDT' in kline_data else ''}+ {len(symbols_to_download)} other symbols")
    
    logger.info(f"Processing {len(symbols_to_download)} symbols in parallel with {max_workers} workers")
    
    # Download data in parallel
    parallel_results = parallel_download_data(
        session=session,
        symbols_info=symbols_to_download,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        category=category,
        max_workers=max_workers
    )
    
    # Merge results
    kline_data.update(parallel_results)
    
    logger.info(f"Completed parallel download: {len(kline_data)} symbols with data")
    
    # Update the summary file with all results
    with open(summary_file, 'a') as f:
        for symbol, df in kline_data.items():
            if df.empty:
                continue
                
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
        
        # Add overall summary
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"Total Symbols Processed: {len(symbols_to_download) + (1 if 'BTCUSDT' in kline_data else 0)}\n")
        f.write(f"Total Symbols with Data: {len(kline_data)}\n")
        f.write(f"Total Records: {sum(len(df) for df in kline_data.values() if not df.empty)}\n")
        
        # Calculate date ranges across all symbols
        if kline_data:
            all_timestamps = []
            for df in kline_data.values():
                if not df.empty:
                    all_timestamps.extend(df['timestamp'].tolist())
            
            if all_timestamps:
                f.write(f"Earliest Data Point: {min(all_timestamps)}\n")
                f.write(f"Latest Data Point: {max(all_timestamps)}\n")
                
                # List symbols with most data
                symbols_by_records = {symbol: len(df) for symbol, df in kline_data.items() if not df.empty}
                top_symbols = sorted(symbols_by_records.items(), key=lambda x: x[1], reverse=True)[:10]
                
                f.write("\nTop 10 Symbols by Number of Records:\n")
                for symbol, count in top_symbols:
                    f.write(f"{symbol}: {count} candles\n")
    
    logger.info(f"Data summary saved to {summary_file}")
    return kline_data
    
# Convenience wrapper functions for futures and spot
def fetch_all_futures_parallel(session, symbols_df, interval="1h", start_date=None, end_date=None, 
                             output_dir="kline_data", max_workers=5, max_symbols=None):
    """Wrapper for fetching futures data in parallel"""
    return fetch_all_data_parallel(
        session=session,
        symbols_df=symbols_df,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        category="linear",
        max_workers=max_workers,
        max_symbols=max_symbols
    )

def fetch_all_spot_parallel(session, symbols_df, interval="1h", start_date=None, end_date=None, 
                           output_dir="spot_data", max_workers=5, max_symbols=None):
    """Wrapper for fetching spot data in parallel"""
    return fetch_all_data_parallel(
        session=session,
        symbols_df=symbols_df,
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        category="spot",
        max_workers=max_workers,
        max_symbols=max_symbols
    )
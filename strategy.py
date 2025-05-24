import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pytz
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from collections import defaultdict
import concurrent.futures

# Import Bybit kline functions for downloading missing data
try:
    from utility.bybit_klines import parallel_download_data, download_kline_data
    from pybit.unified_trading import HTTP
    BYBIT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Bybit kline functions not available: {e}")
    BYBIT_AVAILABLE = False

def try_bitcoin_proxy(symbol, entry_timestamp, kline_dir, start_date, end_date):
    """
    Try to use Bitcoin as a proxy when spot symbol is not available
    
    Args:
        symbol (str): Original symbol that failed
        entry_timestamp (pd.Timestamp): Timestamp for price lookup
        kline_dir (str): Kline data directory
        start_date (str): Start date for potential download
        end_date (str): End date for potential download
        
    Returns:
        tuple: (btc_price, btc_available) where btc_price is the BTC price or None
    """
    print(f"  🪙 Trying Bitcoin as proxy for {symbol}...")
    
    # Try to load Bitcoin spot data
    btc_kline = load_kline_data(
        'BTCUSDT', 'spot', kline_dir,
        auto_download=True,  # Allow download of BTC if missing
        start_date=start_date,
        end_date=end_date,
        launch_time=None  # BTC has been around forever
    )
    
    if not btc_kline.empty:
        try:
            btc_price = get_price_at_timestamp(btc_kline, entry_timestamp)
            if btc_price is not None:
                print(f"  ✅ Bitcoin proxy available @ {btc_price:.2f}")
                return btc_price, True
            else:
                print(f"  ❌ Bitcoin price not available at timestamp {entry_timestamp}")
                return None, False
        except Exception as e:
            print(f"  ❌ Error getting Bitcoin price: {e}")
            return None, False
    else:
        print(f"  ❌ Bitcoin kline data not available")
        return None, False
    
def load_symbol_mapping(mapping_path="future_to_spot_mapping_debug.csv"):
    """
    Load the mapping between futures and spot symbols
    
    Args:
        mapping_path (str): Path to the mapping CSV file
        
    Returns:
        dict: Dictionary mapping future symbol to spot symbol
    """
    try:
        mapping_df = pd.read_csv(mapping_path)
        return dict(zip(mapping_df['symbol'], mapping_df['spotSymbol']))
    except Exception as e:
        print(f"Error loading symbol mapping: {e}")
        return {}

def load_kline_data(symbol, data_type, kline_dir="kline_data", auto_download=True, 
                   start_date=None, end_date=None, launch_time=None):
    """
    Load kline data for a specific symbol and type (linear or spot)
    If data is not found and auto_download is True, attempt to download it
    
    Args:
        symbol (str): Symbol name
        data_type (str): 'linear' or 'spot'
        kline_dir (str): Directory containing kline data
        auto_download (bool): Whether to attempt downloading missing data
        start_date (str): Start date for download if needed
        end_date (str): End date for download if needed
        launch_time (datetime): Launch time of the symbol
        
    Returns:
        pd.DataFrame: Kline data or empty DataFrame if not found
    """
    filename = f"{symbol}_1h_{data_type}_kline.csv"
    filepath = os.path.join(kline_dir, filename)
    
    # Try to load existing file first
    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            if not df.empty:
                # Handle timestamp column carefully
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                    # Standardize timezone handling
                    if df['timestamp'].dt.tz is None:
                        # If no timezone, assume UTC
                        df['timestamp'] = df['timestamp'].dt.tz_localize(pytz.UTC)
                    else:
                        # If has timezone, convert to UTC
                        df['timestamp'] = df['timestamp'].dt.tz_convert(pytz.UTC)
                
                df = df.sort_values('timestamp')
                return df
        except Exception as e:
            print(f"⚠️  Error loading existing file {filepath}: {e}")
    
    # If file doesn't exist or is empty, try to download
    if auto_download and BYBIT_AVAILABLE:
        print(f"📥 Attempting to download missing {data_type} data for {symbol}...")
        
        try:
            # Initialize Bybit session
            session = HTTP()
            
            # Determine category for Bybit API
            category = "linear" if data_type == "linear" else "spot"
            
            # Download the data
            df = download_kline_data(
                session=session,
                symbol=symbol,
                interval="1h",
                start_date=start_date,
                end_date=end_date,
                output_dir=kline_dir,
                category=category,
                launch_time=launch_time
            )
            
            if df is not None and not df.empty:
                # Ensure timezone consistency for downloaded data
                if 'timestamp' in df.columns:
                    if df['timestamp'].dt.tz is None:
                        df['timestamp'] = df['timestamp'].dt.tz_localize(pytz.UTC)
                    else:
                        df['timestamp'] = df['timestamp'].dt.tz_convert(pytz.UTC)
                
                print(f"✅ Successfully downloaded {data_type} data for {symbol}: {len(df)} records")
                return df
            else:
                print(f"❌ Download failed for {symbol} {data_type} - no data returned")
                
        except Exception as e:
            print(f"❌ Error downloading {symbol} {data_type}: {e}")
    
    elif not BYBIT_AVAILABLE and not os.path.exists(filepath):
        print(f"⚠️  File {filepath} not found and Bybit functions not available for auto-download")
    
    return pd.DataFrame()

def get_price_at_timestamp(kline_df, target_timestamp):
    """
    Get the close price at a specific timestamp
    
    Args:
        kline_df (pd.DataFrame): Kline data
        target_timestamp (pd.Timestamp): Target timestamp
        
    Returns:
        float: Close price or None if not found
    """
    if kline_df.empty:
        return None
    
    try:
        # Ensure both timestamps have compatible timezone handling
        df_timestamps = kline_df['timestamp']
        
        # Convert target_timestamp to same timezone format as dataframe
        if target_timestamp.tz is not None and df_timestamps.dt.tz is None:
            # Target has timezone, dataframe doesn't - remove timezone from target
            target_timestamp = target_timestamp.tz_localize(None)

        elif target_timestamp.tz is None and df_timestamps.dt.tz is not None:
            # Target has no timezone, dataframe does - add UTC timezone to target
            target_timestamp = pd.Timestamp(target_timestamp).tz_localize(pytz.UTC)
        
        # Find exact match or closest timestamp
        mask = df_timestamps <= target_timestamp
        if not mask.any():
            return None
        
        closest_row = kline_df[mask].iloc[-1]
        return closest_row['close']
        
    except Exception as e:
        print(f"⚠️  Error getting price at timestamp {target_timestamp}: {e}")
        return None

class Position:
    """Class to track individual trading positions"""
    
    def __init__(self, symbol, position_type, side, units, entry_price, entry_timestamp):
        self.symbol = symbol
        self.position_type = position_type  # 'spot' or 'linear'
        self.side = side  # 'long' or 'short'
        self.units = units
        self.entry_price = entry_price
        self.entry_timestamp = entry_timestamp
        self.exit_price = None
        self.exit_timestamp = None
        self.realized_pnl = 0.0
        self.is_closed = False
    
    def calculate_unrealized_pnl(self, current_price):
        """Calculate unrealized PnL based on current price"""
        if self.is_closed or current_price is None:
            return 0.0
        
        if self.side == 'long':
            return self.units * (current_price - self.entry_price)
        else:  # short
            return self.units * (self.entry_price - current_price)
    
    def close_position(self, exit_price, exit_timestamp):
        """Close the position and calculate realized PnL"""
        if self.is_closed or exit_price is None:
            return 0.0
        
        self.exit_price = exit_price
        self.exit_timestamp = exit_timestamp
        self.is_closed = True
        
        if self.side == 'long':
            self.realized_pnl = self.units * (exit_price - self.entry_price)
        else:  # short
            self.realized_pnl = self.units * (self.entry_price - exit_price)
        
        return self.realized_pnl

def backtest_funding_strategy_with_trading(funding_data, symbols_df=None, output_dir="funding_analysis",
                                         start_date=None, end_date=None, 
                                         initial_capital=1000, min_annual_rate=30.0,
                                         top_n=5, rebalance_days=7,
                                         # Trading cost parameters
                                         taker_fee_pct=0.05,
                                         slippage_pct=0.03,
                                         bid_ask_spread_pct=0.02,
                                         # Data directories
                                         kline_dir="kline_data",
                                         mapping_path="future_to_spot_mapping_debug.csv",
                                         # Auto-download parameters
                                         auto_download=True,
                                         download_batch_size=10):
    """
    Enhanced backtest with actual trading PnL from spot/futures price movements
    FIXED VERSION 3: Added detailed error logging and better handling of missing data
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Create a log file for debugging
    debug_log_path = os.path.join(output_dir, "debug_log.txt")
    debug_log = open(debug_log_path, 'w')
    
    def log_debug(msg):
        """Helper function to log debug messages"""
        with open("debug_log.txt", "a", encoding="utf-8") as debug_log:
            print(msg)
            debug_log.write(f"{datetime.now()}: {msg}\n")
            debug_log.flush()
        
    # Set default dates if not provided
    if start_date is None:
        start_date = "2023-01-01"
        log_debug("⚠️  No start_date provided, using default: 2023-01-01")
    
    if end_date is None:
        end_date = "2025-05-16"
        log_debug("⚠️  No end_date provided, using default: 2025-05-16")
    
    log_debug(f"Starting enhanced backtest from {start_date} to {end_date} with ${initial_capital} initial capital")
    log_debug(f"Strategy: Delta-neutral positions in top {top_n} coins with >={min_annual_rate}% annualized funding rate")
    log_debug(f"Rebalancing every {rebalance_days} days")
    
    if auto_download:
        log_debug(f"🔄 Auto-download enabled: Will attempt to download missing data")
    
    # Load symbol mapping
    symbol_mapping = load_symbol_mapping(mapping_path)
    log_debug(f"Loaded mapping for {len(symbol_mapping)} symbols")
    
    # Convert date strings to datetime objects
    start_dt = pd.to_datetime(start_date).tz_localize(pytz.UTC)
    end_dt = pd.to_datetime(end_date).tz_localize(pytz.UTC)
    
    # Prepare funding data
    all_funding_data = []
    
    def process_symbol_data(symbol_df_tuple):
        symbol, df = symbol_df_tuple
        if df.empty:
            return None
        
        df_copy = df.copy()
        
        if not pd.api.types.is_datetime64_dtype(df_copy['fundingRateTimestamp']):
            df_copy['fundingRateTimestamp'] = pd.to_datetime(df_copy['fundingRateTimestamp'])
        
        if df_copy['fundingRateTimestamp'].dt.tz is None:
            df_copy['fundingRateTimestamp'] = df_copy['fundingRateTimestamp'].dt.tz_localize(pytz.UTC)
        
        df_copy = df_copy[(df_copy['fundingRateTimestamp'] >= start_dt) & 
                          (df_copy['fundingRateTimestamp'] <= end_dt)]
        
        if df_copy.empty:
            return None
        
        df_copy['fundingRate'] = pd.to_numeric(df_copy['fundingRate'], errors='coerce')
        funding_interval_min = df_copy['fundingInterval'].iloc[0]
        funding_events_per_year = 365 * 24 * 60 / funding_interval_min
        df_copy['annualized_rate'] = df_copy['fundingRate'] * funding_events_per_year * 100
        df_copy['symbol'] = symbol
        df_copy['funding_interval_min'] = funding_interval_min
        
        return df_copy
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, os.cpu_count()*4)) as executor:
        future_to_symbol = {executor.submit(process_symbol_data, (symbol, df)): symbol 
                            for symbol, df in funding_data.items()}
        
        for future in concurrent.futures.as_completed(future_to_symbol):
            result = future.result()
            if result is not None:
                all_funding_data.append(result)
    
    if not all_funding_data:
        log_debug("No funding data found for the specified period")
        debug_log.close()
        return None, None, None
    
    combined_df = pd.concat(all_funding_data, ignore_index=True)
    combined_df = combined_df.sort_values('fundingRateTimestamp')
    
    # Create results dataframe
    hours_range = pd.date_range(start=start_dt, end=end_dt, freq='H')
    results_df = pd.DataFrame(index=hours_range)
    results_df.index.name = 'timestamp'
    results_df = results_df.reset_index()
    
    # Initialize tracking columns properly
    results_df['portfolio_value'] = 0.0  # Don't initialize to initial_capital!
    results_df['funding_pnl'] = 0.0
    results_df['trading_pnl_realized'] = 0.0
    results_df['trading_pnl_unrealized'] = 0.0
    results_df['trading_pnl_total'] = 0.0
    results_df['trading_costs'] = 0.0
    results_df['period_return'] = 0.0
    results_df['cumulative_return'] = 0.0
    results_df['num_active_positions'] = 0  # Track number of active positions
    
    # Set only the first row to initial capital
    results_df.loc[0, 'portfolio_value'] = initial_capital
    
    # Trading tracking
    active_positions = {}  # symbol -> {'spot': Position, 'linear': Position}
    position_history = []  # Track all positions for analysis
    data_availability = []  # Track data availability
    missing_data_log = []  # Track when data is missing
    
    # Rebalance tracking
    rebalance_dates = pd.date_range(start=start_dt, end=end_dt, freq=f'{rebalance_days}D')
    rebalance_history = []
    current_capital = initial_capital
    
    # Track cumulative values for proper portfolio calculation
    cumulative_funding_pnl = 0.0
    cumulative_realized_pnl = 0.0
    cumulative_trading_costs = 0.0
    
    # Cache for kline data to avoid repeated loading
    kline_cache = {}
    
    def get_cached_kline(symbol, data_type):
        """Get kline data from cache or load it"""
        cache_key = f"{symbol}_{data_type}"
        if cache_key not in kline_cache:
            kline_cache[cache_key] = load_kline_data(
                symbol, data_type, kline_dir, 
                auto_download=False  # Already downloaded in initial phase
            )
        return kline_cache[cache_key]
    
    log_debug("Running enhanced backtest with trading PnL...")
    
    for i, rebalance_date in enumerate(rebalance_dates):
        # Determine period end
        if i < len(rebalance_dates) - 1:
            next_rebalance = rebalance_dates[i+1]
        else:
            next_rebalance = end_dt
        
        log_debug(f"\nRebalance period {i+1}: {rebalance_date.strftime('%Y-%m-%d')} to {next_rebalance.strftime('%Y-%m-%d')}")
        
        # Get forecast data for symbol selection
        forecast_start = rebalance_date
        forecast_end = rebalance_date + timedelta(days=rebalance_days)
        
        forecast_data = combined_df[
            (combined_df['fundingRateTimestamp'] >= forecast_start) & 
            (combined_df['fundingRateTimestamp'] < forecast_end)
        ]
        
        forecast_rates = forecast_data.groupby('symbol', as_index=False).agg({
            'annualized_rate': 'mean',
            'funding_interval_min': 'first'
        })
        
        forecast_rates = forecast_rates[forecast_rates['annualized_rate'] >= min_annual_rate]
        
        # Track the rebalance trading cost for this period
        period_trading_cost = 0.0
        period_realized_pnl = 0.0
        
        if not forecast_rates.empty:
            forecast_rates = forecast_rates.sort_values('annualized_rate', ascending=False)
            top_symbols = forecast_rates.head(top_n)
            
            # Determine position entry timestamp
            entry_timestamp = rebalance_date
            
            # Check data availability and close old positions
            symbols_to_trade = []
            
            # Close positions that are no longer in the portfolio
            symbols_to_remove = set(active_positions.keys()) - set(top_symbols['symbol'])
            
            for symbol in symbols_to_remove:
                log_debug(f"  Closing positions for {symbol}")
                
                # Get position info
                position_info = active_positions.get(symbol, {})
                is_proxy = position_info.get('is_proxy', False)
                actual_spot_symbol = position_info.get('actual_spot_symbol', symbol_mapping.get(symbol, symbol.replace('USDT', '')))
                
                # Load price data for closing
                linear_kline = get_cached_kline(symbol, 'linear')
                spot_kline = get_cached_kline(actual_spot_symbol, 'spot')
                
                if not linear_kline.empty and not spot_kline.empty:
                    try:
                        linear_exit_price = get_price_at_timestamp(linear_kline, entry_timestamp)
                        spot_exit_price = get_price_at_timestamp(spot_kline, entry_timestamp)
                        
                        if linear_exit_price is not None and spot_exit_price is not None:
                            # Close linear position
                            if 'linear' in active_positions[symbol]:
                                linear_pos = active_positions[symbol]['linear']
                                linear_pnl = linear_pos.close_position(linear_exit_price, entry_timestamp)
                                period_realized_pnl += linear_pnl
                                
                                # Calculate trading costs for closing
                                exit_amount = abs(linear_pos.units * linear_exit_price)
                                cost = exit_amount * (taker_fee_pct + slippage_pct + bid_ask_spread_pct/2) / 100
                                period_trading_cost += cost
                            
                            # Close spot position  
                            if 'spot' in active_positions[symbol]:
                                spot_pos = active_positions[symbol]['spot']
                                spot_pnl = spot_pos.close_position(spot_exit_price, entry_timestamp)
                                period_realized_pnl += spot_pnl
                                
                                # Calculate trading costs for closing
                                exit_amount = abs(spot_pos.units * spot_exit_price)
                                cost = exit_amount * (taker_fee_pct + slippage_pct + bid_ask_spread_pct/2) / 100
                                period_trading_cost += cost
                            
                            proxy_msg = f" (was using {actual_spot_symbol} as proxy)" if is_proxy else ""
                            log_debug(f"    ✅ Closed positions for {symbol}{proxy_msg}, PnL: ${linear_pnl + spot_pnl:.2f}")
                        else:
                            log_debug(f"    ❌ Could not get exit prices for {symbol}")
                    except Exception as e:
                        log_debug(f"⚠️  Error closing positions for {symbol}: {e}")
                else:
                    log_debug(f"    ❌ Missing kline data for closing {symbol}")
                
                # Remove from active positions
                if symbol in active_positions:
                    del active_positions[symbol]
            
            # Check new symbols for data availability
            for _, row in top_symbols.iterrows():
                symbol = row['symbol']
                spot_symbol = symbol_mapping.get(symbol, symbol.replace('USDT', ''))
                
                # Skip if already have position
                if symbol in active_positions:
                    continue
                
                # Get launch time for this symbol
                launch_time = None
                if symbols_df is not None:
                    symbol_row = symbols_df[symbols_df['symbol'] == symbol]
                    if not symbol_row.empty and 'launchTime' in symbol_row.columns:
                        launch_time = symbol_row['launchTime'].iloc[0]
                
                # Load kline data (with auto-download if enabled)
                linear_kline = load_kline_data(
                    symbol, 'linear', kline_dir, 
                    auto_download=auto_download,
                    start_date=start_date,
                    end_date=end_date,
                    launch_time=launch_time
                )
                
                spot_kline = load_kline_data(
                    spot_symbol, 'spot', kline_dir,
                    auto_download=auto_download, 
                    start_date=start_date,
                    end_date=end_date,
                    launch_time=launch_time
                )
                
                # Cache the data
                kline_cache[f"{symbol}_linear"] = linear_kline
                kline_cache[f"{spot_symbol}_spot"] = spot_kline
                
                # Check if both datasets are available
                linear_available = not linear_kline.empty
                spot_available = not spot_kline.empty
                both_available = linear_available and spot_available
                
                # Get prices at entry timestamp
                linear_price = None
                spot_price = None
                
                try:
                    if linear_available:
                        linear_price = get_price_at_timestamp(linear_kline, entry_timestamp)
                    if spot_available:
                        spot_price = get_price_at_timestamp(spot_kline, entry_timestamp)
                except Exception as e:
                    log_debug(f"⚠️  Error getting prices for {symbol}: {e}")
                
                prices_available = (linear_price is not None) and (spot_price is not None)
                
                # Record data availability
                data_availability.append({
                    'rebalance_date': rebalance_date,
                    'symbol': symbol,
                    'spot_symbol': spot_symbol,
                    'linear_data_available': linear_available,
                    'spot_data_available': spot_available,
                    'both_available': both_available,
                    'linear_price_available': linear_price is not None,
                    'spot_price_available': spot_price is not None,
                    'prices_available': prices_available,
                    'entry_timestamp': entry_timestamp,
                    'linear_price': linear_price,
                    'spot_price': spot_price,
                    'avg_funding_rate': row['annualized_rate'],
                    'bitcoin_proxy_used': False,
                    'bitcoin_proxy_attempted': False
                })
                
                if both_available and prices_available:
                    symbols_to_trade.append({
                        'symbol': symbol,
                        'spot_symbol': spot_symbol,
                        'linear_price': linear_price,
                        'spot_price': spot_price,
                        'funding_rate': row['annualized_rate'],
                        'is_proxy': False,
                        'actual_spot_symbol': spot_symbol
                    })
                # Skip Bitcoin proxy - it's causing problems
                else:
                    status_msg = f"Linear: {linear_available}, Spot: {spot_available}, Prices: {prices_available}"
                    log_debug(f"  ⚠️  Skipping {symbol} -> {spot_symbol}: {status_msg}")
            
            # Calculate allocation and open new positions
            if symbols_to_trade:
                # Calculate allocation based on current portfolio value minus costs
                allocation_per_symbol = (current_capital - period_trading_cost) / len(symbols_to_trade)
                
                for symbol_info in symbols_to_trade:
                    symbol = symbol_info['symbol']
                    spot_symbol = symbol_info['spot_symbol']
                    linear_price = symbol_info['linear_price']
                    spot_price = symbol_info['spot_price']
                    avg_funding_rate = symbol_info['funding_rate']
                    
                    is_proxy = symbol_info.get('is_proxy', False)
                    actual_spot_symbol = symbol_info.get('actual_spot_symbol', spot_symbol)
                    
                    # Determine position direction based on funding rate
                    if avg_funding_rate > 0:
                        linear_side = 'short'
                        spot_side = 'long'
                    else:
                        linear_side = 'long'
                        spot_side = 'short'
                    
                    # Calculate units for each position (delta-neutral)
                    position_size_usd = allocation_per_symbol / 2  # Half for each leg
                    linear_units = position_size_usd / linear_price
                    spot_units = position_size_usd / spot_price
                    
                    # Calculate trading costs for opening positions
                    linear_cost = position_size_usd * (taker_fee_pct + slippage_pct + bid_ask_spread_pct/2) / 100
                    spot_cost = position_size_usd * (taker_fee_pct + slippage_pct + bid_ask_spread_pct/2) / 100
                    period_trading_cost += (linear_cost + spot_cost)
                    
                    # Create positions
                    linear_position = Position(symbol, 'linear', linear_side, linear_units, linear_price, entry_timestamp)
                    spot_position = Position(actual_spot_symbol, 'spot', spot_side, spot_units, spot_price, entry_timestamp)
                    
                    # Store positions with metadata
                    active_positions[symbol] = {
                        'linear': linear_position,
                        'spot': spot_position,
                        'is_proxy': is_proxy,
                        'original_spot_symbol': spot_symbol,
                        'actual_spot_symbol': actual_spot_symbol,
                        'allocation': allocation_per_symbol
                    }
                    
                    position_history.extend([linear_position, spot_position])
                    
                    proxy_msg = f" (using {actual_spot_symbol} as proxy)" if is_proxy else ""
                    log_debug(f"  ✅ Opened delta-neutral position for {symbol}: {linear_side} linear @ {linear_price:.6f}, {spot_side} spot @ {spot_price:.6f}{proxy_msg}")
            
            # Update cumulative values
            cumulative_trading_costs += period_trading_cost
            cumulative_realized_pnl += period_realized_pnl
            
            # Record rebalance event
            rebalance_event = {
                'date': rebalance_date,
                'symbols_selected': len(symbols_to_trade),
                'symbols_with_data': len([s for s in data_availability if s['rebalance_date'] == rebalance_date and s['both_available']]),
                'total_symbols_evaluated': len(top_symbols),
                'trading_costs': period_trading_cost,
                'realized_pnl': period_realized_pnl,
                'portfolio_value': current_capital,
                'active_positions': len(active_positions),
                'active_symbols': list(active_positions.keys())
            }
            rebalance_history.append(rebalance_event)
            
            log_debug(f"  Portfolio update: {len(symbols_to_trade)} positions, ${period_trading_cost:.2f} trading costs, ${period_realized_pnl:.2f} realized PnL")
        
        else:
            log_debug("  No symbols meet funding rate criteria - closing all positions")
            # Close all positions if no symbols qualify
            for symbol in list(active_positions.keys()):
                log_debug(f"  Closing positions for {symbol} (no funding rate criteria met)")
                
                # Try to close with proper prices
                position_info = active_positions.get(symbol, {})
                actual_spot_symbol = position_info.get('actual_spot_symbol', symbol_mapping.get(symbol, symbol.replace('USDT', '')))
                
                linear_kline = get_cached_kline(symbol, 'linear')
                spot_kline = get_cached_kline(actual_spot_symbol, 'spot')
                
                if not linear_kline.empty and not spot_kline.empty:
                    try:
                        linear_exit_price = get_price_at_timestamp(linear_kline, rebalance_date)
                        spot_exit_price = get_price_at_timestamp(spot_kline, rebalance_date)
                        
                        if linear_exit_price is not None and spot_exit_price is not None:
                            # Close positions and record PnL
                            if 'linear' in active_positions[symbol]:
                                linear_pos = active_positions[symbol]['linear']
                                linear_pnl = linear_pos.close_position(linear_exit_price, rebalance_date)
                                period_realized_pnl += linear_pnl
                            
                            if 'spot' in active_positions[symbol]:
                                spot_pos = active_positions[symbol]['spot']
                                spot_pnl = spot_pos.close_position(spot_exit_price, rebalance_date)
                                period_realized_pnl += spot_pnl
                    except Exception as e:
                        log_debug(f"    Error closing {symbol}: {e}")
                
                if symbol in active_positions:
                    del active_positions[symbol]
            
            # Update cumulative realized PnL
            cumulative_realized_pnl += period_realized_pnl
        
        # Process each hour in this period to calculate PnL
        period_hours = results_df[
            (results_df['timestamp'] >= rebalance_date) & 
            (results_df['timestamp'] < next_rebalance)
        ].index.tolist()
        
        # Track if we've recorded the rebalance cost for this period
        rebalance_cost_recorded = False
        
        for hour_idx in period_hours:
            hour_timestamp = results_df.loc[hour_idx, 'timestamp']
            
            # Record trading costs and realized PnL at the rebalance hour
            if hour_timestamp >= rebalance_date and not rebalance_cost_recorded:
                if period_trading_cost > 0:
                    results_df.loc[hour_idx, 'trading_costs'] = -period_trading_cost  # Negative because it's a cost
                if period_realized_pnl != 0:
                    results_df.loc[hour_idx, 'trading_pnl_realized'] = period_realized_pnl
                rebalance_cost_recorded = True
            
            # Calculate funding PnL
            hour_funding_pnl = 0.0
            
            # Check each symbol in portfolio for funding payments
            for symbol, positions in active_positions.items():
                # Find if there's a funding payment at this hour
                symbol_funding = combined_df[
                    (combined_df['symbol'] == symbol) & 
                    (combined_df['fundingRateTimestamp'] == hour_timestamp)
                ]
                
                if not symbol_funding.empty and 'allocation' in positions:
                    # Calculate funding payment based on position
                    funding_rate = symbol_funding['fundingRate'].iloc[0]
                    
                    # Get the position value
                    position_value = positions['allocation']
                    
                    # For short linear positions, we collect positive funding
                    # For long linear positions, we pay positive funding
                    if positions['linear'].side == 'short':
                        funding_payment = position_value * funding_rate
                    else:
                        funding_payment = -position_value * funding_rate
                    
                    hour_funding_pnl += funding_payment
            
            # Calculate trading PnL (unrealized) for delta-neutral positions
            hour_unrealized_pnl = 0.0
            positions_with_data = 0
            positions_missing_data = 0
            
            for symbol, positions in active_positions.items():
                # Get position info
                actual_spot_symbol = positions.get('actual_spot_symbol', symbol_mapping.get(symbol, symbol.replace('USDT', '')))
                
                # Load current prices
                linear_kline = get_cached_kline(symbol, 'linear')
                spot_kline = get_cached_kline(actual_spot_symbol, 'spot')
                
                if not linear_kline.empty and not spot_kline.empty:
                    try:
                        linear_current_price = get_price_at_timestamp(linear_kline, hour_timestamp)
                        spot_current_price = get_price_at_timestamp(spot_kline, hour_timestamp)
                        
                        if linear_current_price is not None and spot_current_price is not None:
                            # Calculate PnL for both legs
                            linear_pnl = positions['linear'].calculate_unrealized_pnl(linear_current_price)
                            spot_pnl = positions['spot'].calculate_unrealized_pnl(spot_current_price)
                            
                            # Total PnL for this delta-neutral position
                            position_pnl = linear_pnl + spot_pnl
                            hour_unrealized_pnl += position_pnl
                            positions_with_data += 1
                        else:
                            positions_missing_data += 1
                            missing_data_log.append({
                                'timestamp': hour_timestamp,
                                'symbol': symbol,
                                'reason': 'Price not found at timestamp',
                                'linear_kline_empty': linear_kline.empty,
                                'spot_kline_empty': spot_kline.empty,
                                'linear_price': linear_current_price,
                                'spot_price': spot_current_price
                            })
                    except Exception as e:
                        # LOG THE ERROR INSTEAD OF SILENTLY IGNORING IT
                        positions_missing_data += 1
                        error_msg = f"ERROR calculating PnL at {hour_timestamp} for {symbol}: {str(e)}"
                        log_debug(error_msg)
                        missing_data_log.append({
                            'timestamp': hour_timestamp,
                            'symbol': symbol,
                            'reason': f'Exception: {str(e)}',
                            'linear_kline_empty': linear_kline.empty,
                            'spot_kline_empty': spot_kline.empty
                        })
                else:
                    positions_missing_data += 1
                    missing_data_log.append({
                        'timestamp': hour_timestamp,
                        'symbol': symbol,
                        'reason': 'Kline data empty',
                        'linear_kline_empty': linear_kline.empty,
                        'spot_kline_empty': spot_kline.empty
                    })
            
            # Log if we're missing data for positions
            if positions_missing_data > 0 and len(active_positions) > 0:
                log_debug(f"  ⚠️  {hour_timestamp}: Missing data for {positions_missing_data}/{len(active_positions)} positions")
            
            # Update cumulative values
            cumulative_funding_pnl += hour_funding_pnl
            
            # Update results
            results_df.loc[hour_idx, 'funding_pnl'] = hour_funding_pnl
            results_df.loc[hour_idx, 'trading_pnl_unrealized'] = hour_unrealized_pnl
            results_df.loc[hour_idx, 'trading_pnl_total'] = hour_unrealized_pnl + cumulative_realized_pnl
            results_df.loc[hour_idx, 'num_active_positions'] = len(active_positions)
            
            # Calculate portfolio value correctly
            # Portfolio value = Initial capital + cumulative funding PnL + unrealized PnL + realized PnL - cumulative costs
            portfolio_value = initial_capital + cumulative_funding_pnl + hour_unrealized_pnl + cumulative_realized_pnl - cumulative_trading_costs
            results_df.loc[hour_idx, 'portfolio_value'] = portfolio_value
            
        # At the end of each rebalance period, update current_capital for next rebalance
        if len(period_hours) > 0:
            last_hour_idx = period_hours[-1] 
            current_capital = results_df.loc[last_hour_idx, 'portfolio_value']
    
    # Close debug log
    debug_log.close()
    
    # Save missing data log
    if missing_data_log:
        missing_data_df = pd.DataFrame(missing_data_log)
        missing_data_path = os.path.join(output_dir, "missing_data_log.csv")
        missing_data_df.to_csv(missing_data_path, index=False)
        log_debug(f"Missing data log saved to {missing_data_path}")
    
    # Calculate final metrics and save results (rest of the code remains the same)
    print("\nCalculating final metrics...")
    
    # Calculate performance metrics
    results_df['period_return'] = results_df['portfolio_value'].pct_change()
    results_df.loc[0, 'period_return'] = (results_df.loc[0, 'portfolio_value'] - initial_capital) / initial_capital
    results_df['period_return'] = results_df['period_return'].fillna(0).replace([np.inf, -np.inf], 0)
    results_df['cumulative_return'] = (results_df['portfolio_value'] / initial_capital) - 1
    
    # Add cumulative columns for plotting
    results_df['cum_funding_pnl'] = results_df['funding_pnl'].cumsum()
    results_df['cum_trading_pnl'] = results_df['trading_pnl_total']  # Already cumulative
    results_df['cum_trading_costs'] = results_df['trading_costs'].cumsum()
    
    # Performance summary
    total_days = (end_dt - start_dt).days
    total_return = results_df['portfolio_value'].iloc[-1] / initial_capital - 1
    annual_return = ((1 + total_return) ** (365 / total_days)) - 1
    
    daily_returns = results_df.set_index('timestamp').resample('D')['period_return'].sum()
    volatility = daily_returns.std() * np.sqrt(365)
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    
    results_df['peak'] = results_df['portfolio_value'].cummax()
    results_df['drawdown'] = (results_df['portfolio_value'] - results_df['peak']) / results_df['peak']
    max_drawdown = results_df['drawdown'].min()
    
    # Calculate total costs and PnL
    total_trading_costs = abs(results_df['cum_trading_costs'].iloc[-1])
    total_funding_pnl = results_df['cum_funding_pnl'].iloc[-1]
    total_trading_pnl = results_df['cum_trading_pnl'].iloc[-1]
    total_realized_pnl = sum(pos.realized_pnl for pos in position_history if pos.is_closed)
    
    print(f"\n=== ENHANCED BACKTEST RESULTS ===")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annual_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Max Drawdown: {max_drawdown:.2%}")
    print(f"Final Portfolio Value: ${results_df['portfolio_value'].iloc[-1]:.2f}")
    print(f"Total Funding PnL: ${total_funding_pnl:.2f}")
    print(f"Total Trading PnL: ${total_trading_pnl:.2f}")
    print(f"Total Trading Costs: ${total_trading_costs:.2f}")
    print(f"Total Realized Trading PnL: ${total_realized_pnl:.2f}")
    print(f"Data availability: {len([d for d in data_availability if d['both_available']])}/{len(data_availability)} symbols had both spot and linear data")
    
    if auto_download:
        attempted_downloads = len([d for d in data_availability if d.get('auto_download_attempted', False)])
        successful_downloads = len([d for d in data_availability if d.get('download_success', False)])
        print(f"Auto-download summary: {successful_downloads}/{attempted_downloads} successful downloads")

    if data_availability:
        proxy_used = len([d for d in data_availability if d.get('bitcoin_proxy_used', False)])
        if proxy_used > 0:
            print(f"Bitcoin proxy used for {proxy_used} symbols")
    
    # Save results
    results_path = f"{output_dir}/enhanced_funding_strategy_results.csv"
    results_df.to_csv(results_path, index=False)
    
    # Save data availability report
    availability_path = f"{output_dir}/data_availability_report.csv"
    pd.DataFrame(data_availability).to_csv(availability_path, index=False)
    
    # Save position tracking
    position_data = []
    for pos in position_history:
        position_data.append({
            'symbol': pos.symbol,
            'type': pos.position_type,
            'side': pos.side,
            'units': pos.units,
            'entry_price': pos.entry_price,
            'entry_timestamp': pos.entry_timestamp,
            'exit_price': pos.exit_price,
            'exit_timestamp': pos.exit_timestamp,
            'realized_pnl': pos.realized_pnl,
            'is_closed': pos.is_closed
        })
    
    positions_path = f"{output_dir}/position_history.csv"
    pd.DataFrame(position_data).to_csv(positions_path, index=False)
    
    print(f"\nResults saved:")
    print(f"  - Main results: {results_path}")
    print(f"  - Data availability: {availability_path}")
    print(f"  - Position history: {positions_path}")
    
    # Plot results
    plot_enhanced_backtest_results(results_df[:-1], rebalance_history, output_dir, initial_capital)
    
    return results_df, rebalance_history, data_availability

def plot_enhanced_backtest_results(results_df, rebalance_history, output_dir, initial_capital):
    """
    Plot enhanced backtest results with trading PnL breakdown
    """
    daily_df = results_df.set_index('timestamp').resample('D').last().reset_index()
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(15, 16), sharex=True)
    
    # Plot 1: Portfolio Value
    axes[0].plot(daily_df['timestamp'], daily_df['portfolio_value'], 'b-', linewidth=1.5)
    axes[0].axhline(y=initial_capital, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
    axes[0].set_title('Enhanced Portfolio Value Over Time', fontsize=14)
    axes[0].set_ylabel('Portfolio Value (USD)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: PnL Components
    axes[1].plot(daily_df['timestamp'], daily_df['cum_funding_pnl'], 'g-', linewidth=1.5, label='Funding PnL')
    axes[1].plot(daily_df['timestamp'], daily_df['cum_trading_pnl'], 'b-', linewidth=1.5, label='Trading PnL')
    axes[1].plot(daily_df['timestamp'], daily_df['cum_trading_costs'], 'r-', linewidth=1.5, label='Trading Costs')
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.2)
    axes[1].set_title('PnL Components Breakdown', fontsize=14)
    axes[1].set_ylabel('Cumulative PnL (USD)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot 3: Returns
    axes[2].plot(daily_df['timestamp'], daily_df['cumulative_return'] * 100, 'purple', linewidth=1.5)
    axes[2].set_title('Cumulative Return (%)', fontsize=14)
    axes[2].set_ylabel('Return (%)', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Drawdowns
    axes[3].fill_between(daily_df['timestamp'], daily_df['drawdown'] * 100, 0, color='r', alpha=0.3)
    axes[3].set_title('Drawdowns (%)', fontsize=14)
    axes[3].set_ylabel('Drawdown (%)', fontsize=12)
    axes[3].set_xlabel('Date', fontsize=12)
    axes[3].grid(True, alpha=0.3)
    axes[3].invert_yaxis()
    
    # Format dates
    plt.gcf().autofmt_xdate()
    
    # Add performance metrics
    total_return = daily_df['portfolio_value'].iloc[-1] / initial_capital - 1
    total_funding_pnl = daily_df['cum_funding_pnl'].iloc[-1]
    total_trading_pnl = daily_df['cum_trading_pnl'].iloc[-1]
    total_costs = abs(daily_df['cum_trading_costs'].iloc[-1])
    
    metrics_text = (
        f"Performance Summary:\n"
        f"Total Return: {total_return:.2%}\n"
        f"Final Value: ${daily_df['portfolio_value'].iloc[-1]:.2f}\n"
        f"Funding PnL: ${total_funding_pnl:.2f}\n"
        f"Trading PnL: ${total_trading_pnl:.2f}\n"
        f"Trading Costs: ${total_costs:.2f}"
    )
    
    fig.text(0.02, 0.02, metrics_text, fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save figure
    output_path = f"{output_dir}/enhanced_funding_strategy_performance.png"
    plt.savefig(output_path, dpi=300)
    print(f"Enhanced performance chart saved to {output_path}")
    
    plt.show()

# Additional utility function to analyze data availability
def analyze_data_availability(availability_data):
    """
    Analyze and report on data availability issues including Bitcoin proxy usage
    """
    df = pd.DataFrame(availability_data)
    
    print("\n=== DATA AVAILABILITY ANALYSIS ===")
    print(f"Total symbol evaluations: {len(df)}")
    print(f"Both spot and linear data available: {df['both_available'].sum()} ({df['both_available'].mean()*100:.1f}%)")
    print(f"Linear data available: {df['linear_data_available'].sum()} ({df['linear_data_available'].mean()*100:.1f}%)")
    print(f"Spot data available: {df['spot_data_available'].sum()} ({df['spot_data_available'].mean()*100:.1f}%)")
    print(f"Both prices available: {df['prices_available'].sum()} ({df['prices_available'].mean()*100:.1f}%)")
    
    # Bitcoin proxy statistics
    if 'bitcoin_proxy_attempted' in df.columns:
        proxy_attempted = df['bitcoin_proxy_attempted'].sum()
        proxy_successful = df['bitcoin_proxy_used'].sum()
        print(f"\n=== BITCOIN PROXY STATISTICS ===")
        print(f"Bitcoin proxy attempts: {proxy_attempted}")
        print(f"Bitcoin proxy successful: {proxy_successful}")
        if proxy_attempted > 0:
            print(f"Bitcoin proxy success rate: {proxy_successful/proxy_attempted*100:.1f}%")
        
        # Show symbols that used Bitcoin proxy
        if 'bitcoin_proxy_used' in df.columns:
            proxy_symbols = df[df['bitcoin_proxy_used'] == True]['symbol'].unique()
            if len(proxy_symbols) > 0:
                print(f"Symbols using Bitcoin proxy ({len(proxy_symbols)}):")
                for symbol in proxy_symbols[:10]:  # Show first 10
                    print(f"  - {symbol}")
                if len(proxy_symbols) > 10:
                    print(f"  ... and {len(proxy_symbols) - 10} more")
    
    # Missing data summary
    missing_linear = df[~df['linear_data_available']]['symbol'].unique()
    missing_spot = df[~df['spot_data_available']]['symbol'].unique()
    
    if len(missing_linear) > 0:
        print(f"\nMissing LINEAR data for {len(missing_linear)} symbols:")
        print(", ".join(missing_linear[:10]) + ("..." if len(missing_linear) > 10 else ""))
    
    if len(missing_spot) > 0:
        print(f"\nMissing SPOT data for {len(missing_spot)} symbols:")
        print(", ".join(missing_spot[:10]) + ("..." if len(missing_spot) > 10 else ""))
    
    # Symbols that were consistently missing data
    symbol_availability = df.groupby('symbol').agg({
        'both_available': 'mean',
        'linear_data_available': 'mean', 
        'spot_data_available': 'mean',
        'bitcoin_proxy_used': 'any' if 'bitcoin_proxy_used' in df.columns else lambda x: False
    }).reset_index()
    
    never_available = symbol_availability[symbol_availability['both_available'] == 0]
    if len(never_available) > 0:
        print(f"\nSymbols NEVER having both datasets ({len(never_available)} symbols):")
        for _, row in never_available.head(10).iterrows():
            proxy_used = " (used BTC proxy)" if row.get('bitcoin_proxy_used', False) else ""
            print(f"  {row['symbol']}: Linear={row['linear_data_available']:.0%}, Spot={row['spot_data_available']:.0%}{proxy_used}")
    
    # Overall effectiveness including proxy
    if 'bitcoin_proxy_used' in df.columns:
        total_effective = df['both_available'].sum() + df['bitcoin_proxy_used'].sum()
        effectiveness_rate = total_effective / len(df) * 100
        print(f"\nOVERALL EFFECTIVENESS (including Bitcoin proxy): {effectiveness_rate:.1f}%")
        print(f"  - Direct spot/linear pairs: {df['both_available'].sum()}")
        print(f"  - Bitcoin proxy pairs: {df['bitcoin_proxy_used'].sum()}")
        print(f"  - Total tradeable: {total_effective}")
    
    return df
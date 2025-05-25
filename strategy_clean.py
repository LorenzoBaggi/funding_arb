import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pytz
from datetime import datetime, timedelta
import concurrent.futures
from collections import defaultdict

class VolatilityScaler:
    """Handles volatility scaling calculations and tracking"""
    
    def __init__(self, target_vol=0.15, ewma_lambda=0.06):
        self.target_vol = target_vol
        self.ewma_lambda = ewma_lambda
        self.ewmas = [0.10**2]  # Initialize with 10% annualized vol squared
        self.ewstrats = [1.0]   # Initialize scalar at 1.0
    
    def update_volatility_scaling(self, daily_return):
        """Update EWMA variance and calculate new volatility scalar"""
        # Update EWMA of variance
        daily_variance = daily_return**2
        new_ewma = self.ewma_lambda * daily_variance + (1 - self.ewma_lambda) * self.ewmas[-1]
        self.ewmas.append(new_ewma)
        
        # Calculate annualized realized volatility
        ann_realized_vol = np.sqrt(new_ewma * 365)
        
        # Calculate new volatility scalar
        if ann_realized_vol > 0:
            new_scalar = self.ewstrats[-1] * self.target_vol / ann_realized_vol
        else:
            new_scalar = self.ewstrats[-1]
        
        # Update EWMA of scalar
        scalar_ewma = self.ewma_lambda * new_scalar + (1 - self.ewma_lambda) * self.ewstrats[-1]
        self.ewstrats.append(scalar_ewma)
        
        return ann_realized_vol, scalar_ewma
    
    def get_current_metrics(self):
        """Get current volatility metrics"""
        return {
            'realized_vol': np.sqrt(self.ewmas[-1] * 365),
            'vol_scalar': self.ewstrats[-1],
            'target_vol': self.target_vol
        }

class BacktestLogger:
    """Handles verbose logging for the backtest"""
    
    def __init__(self, output_dir):
        self.debug_log_path = os.path.join(output_dir, "verbose_vol_scaled_backtest_log.txt")
        self._initialize_log()
    
    def _initialize_log(self):
        """Initialize the log file"""
        with open(self.debug_log_path, 'w', encoding="utf-8") as f:
            f.write(f"=== VERBOSE VOLATILITY SCALED FUNDING STRATEGY BACKTEST LOG ===\n")
            f.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    def log(self, msg):
        """Log a message to both console and file"""
        print(msg)
        with open(self.debug_log_path, "a", encoding="utf-8") as debug_log:
            debug_log.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {msg}\n")
            debug_log.flush()

class PositionManager:
    """Manages positions and trading logic"""
    
    def __init__(self, symbol_mapping, taker_fee_pct, slippage_pct, bid_ask_spread_pct):
        self.symbol_mapping = symbol_mapping
        self.taker_fee_pct = taker_fee_pct
        self.slippage_pct = slippage_pct
        self.bid_ask_spread_pct = bid_ask_spread_pct
        self.active_positions = {}
        self.position_history = []
    
    def calculate_trading_cost(self, position_size_usd):
        """Calculate trading costs for a position"""
        return position_size_usd * (self.taker_fee_pct + self.slippage_pct + self.bid_ask_spread_pct/2) / 100
    
    def open_position(self, symbol, spot_symbol, linear_price, spot_price, 
                      scaled_allocation, avg_funding_rate, entry_timestamp, 
                      base_allocation, vol_scalar, is_proxy=False):
        """Open a new delta-neutral position"""
        
        # Determine position direction based on funding rate
        if avg_funding_rate > 0:
            linear_side = 'short'
            spot_side = 'long'
        else:
            linear_side = 'long'
            spot_side = 'short'
        
        # Calculate units for each position (delta-neutral)
        position_size_usd = scaled_allocation / 2  # Half for each leg
        linear_units = position_size_usd / linear_price
        spot_units = position_size_usd / spot_price
        
        # Calculate trading costs
        linear_cost = self.calculate_trading_cost(position_size_usd)
        spot_cost = self.calculate_trading_cost(position_size_usd)
        total_cost = linear_cost + spot_cost
        
        # Create positions
        linear_position = Position(symbol, 'linear', linear_side, linear_units, linear_price, entry_timestamp)
        spot_position = Position(spot_symbol, 'spot', spot_side, spot_units, spot_price, entry_timestamp)
        
        # Store positions with metadata
        self.active_positions[symbol] = {
            'linear': linear_position,
            'spot': spot_position,
            'is_proxy': is_proxy,
            'original_spot_symbol': spot_symbol,
            'actual_spot_symbol': spot_symbol,
            'base_allocation': base_allocation,
            'scaled_allocation': scaled_allocation,
            'allocation': scaled_allocation,
            'vol_scalar_at_entry': vol_scalar
        }
        
        self.position_history.extend([linear_position, spot_position])
        
        return total_cost, linear_cost, spot_cost
    
    def close_position(self, symbol, linear_price, spot_price, exit_timestamp):
        """Close a position and return PnL and costs"""
        if symbol not in self.active_positions:
            return 0.0, 0.0, 0.0  # pnl, cost, realized_pnl
        
        positions = self.active_positions[symbol]
        total_pnl = 0.0
        total_cost = 0.0
        
        # Close linear position
        if 'linear' in positions:
            linear_pos = positions['linear']
            linear_pnl = linear_pos.close_position(linear_price, exit_timestamp)
            total_pnl += linear_pnl
            
            exit_amount = abs(linear_pos.units * linear_price)
            cost = self.calculate_trading_cost(exit_amount)
            total_cost += cost
        
        # Close spot position
        if 'spot' in positions:
            spot_pos = positions['spot']
            spot_pnl = spot_pos.close_position(spot_price, exit_timestamp)
            total_pnl += spot_pnl
            
            exit_amount = abs(spot_pos.units * spot_price)
            cost = self.calculate_trading_cost(exit_amount)
            total_cost += cost
        
        # Remove from active positions
        del self.active_positions[symbol]
        
        return total_pnl, total_cost, total_pnl
    
    def update_position_allocations(self, current_base_portfolio, vol_scalar):
        """Update all active positions with new vol scaling"""
        for symbol, positions in self.active_positions.items():
            if symbol in current_base_portfolio:
                base_allocation = current_base_portfolio[symbol]
                new_scaled_allocation = base_allocation * vol_scalar
                
                positions['scaled_allocation'] = new_scaled_allocation
                positions['allocation'] = new_scaled_allocation

class DataManager:
    """Handles data loading and caching"""
    
    def __init__(self, kline_dir, auto_download=True):
        self.kline_dir = kline_dir
        self.auto_download = auto_download
        self.kline_cache = {}
    
    def get_cached_kline(self, symbol, data_type):
        """Get kline data from cache or load it"""
        cache_key = f"{symbol}_{data_type}"
        if cache_key not in self.kline_cache:
            self.kline_cache[cache_key] = load_kline_data(
                symbol, data_type, self.kline_dir, 
                auto_download=False
            )
        return self.kline_cache[cache_key]
    
    def load_symbol_data(self, symbol, spot_symbol, start_date, end_date, launch_time=None):
        """Load both linear and spot data for a symbol"""
        linear_kline = load_kline_data(
            symbol, 'linear', self.kline_dir, 
            auto_download=self.auto_download,
            start_date=start_date,
            end_date=end_date,
            launch_time=launch_time
        )
        
        spot_kline = load_kline_data(
            spot_symbol, 'spot', self.kline_dir,
            auto_download=self.auto_download, 
            start_date=start_date,
            end_date=end_date,
            launch_time=launch_time
        )
        
        # Cache the data
        self.kline_cache[f"{symbol}_linear"] = linear_kline
        self.kline_cache[f"{spot_symbol}_spot"] = spot_kline
        
        return linear_kline, spot_kline
    
    def try_bitcoin_proxy(self, symbol, entry_timestamp, start_date, end_date, logger):
        """Try to use Bitcoin as a proxy when spot symbol is not available"""
        logger.log(f"    🪙 Trying Bitcoin as proxy for {symbol}...")
        
        btc_kline = load_kline_data(
            'BTCUSDT', 'spot', self.kline_dir,
            auto_download=self.auto_download,
            start_date=start_date,
            end_date=end_date,
            launch_time=None
        )
        
        self.kline_cache["BTCUSDT_spot"] = btc_kline
        
        if not btc_kline.empty:
            try:
                btc_price = get_price_at_timestamp(btc_kline, entry_timestamp)
                if btc_price is not None:
                    logger.log(f"    ✅ Bitcoin proxy available @ {btc_price:.2f}")
                    return btc_price, True
                else:
                    logger.log(f"    ❌ Bitcoin price not available at timestamp {entry_timestamp}")
                    return None, False
            except Exception as e:
                logger.log(f"    ❌ Error getting Bitcoin price: {e}")
                return None, False
        else:
            logger.log(f"    ❌ Bitcoin kline data not available")
            return None, False

def validate_dependencies():
    """Check if required functions exist"""
    required_functions = ['load_symbol_mapping', 'load_kline_data', 'get_price_at_timestamp', 'Position']
    missing_functions = []
    
    for func_name in required_functions:
        if func_name not in globals():
            missing_functions.append(func_name)
    
    return missing_functions

def process_funding_data(funding_data, start_dt, end_dt):
    """Process and prepare funding data for backtesting"""
    
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
    
    # Process data in parallel
    all_funding_data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, os.cpu_count()*4)) as executor:
        future_to_symbol = {executor.submit(process_symbol_data, (symbol, df)): symbol 
                            for symbol, df in funding_data.items()}
        
        for future in concurrent.futures.as_completed(future_to_symbol):
            result = future.result()
            if result is not None:
                all_funding_data.append(result)
    
    if not all_funding_data:
        return None
    
    combined_df = pd.concat(all_funding_data, ignore_index=True)
    return combined_df.sort_values('fundingRateTimestamp')

def select_trading_symbols(combined_df, rebalance_date, rebalance_days, min_annual_rate, top_n):
    """Select symbols for trading based on funding rates"""
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
    
    if not forecast_rates.empty:
        forecast_rates = forecast_rates.sort_values('annualized_rate', ascending=False)
        return forecast_rates.head(top_n)
    
    return pd.DataFrame()

def calculate_funding_pnl(active_positions, combined_df, hour_timestamp):
    """Calculate funding PnL for the current hour"""
    hour_funding_pnl = 0.0
    
    for symbol, positions in active_positions.items():
        symbol_funding = combined_df[
            (combined_df['symbol'] == symbol) & 
            (combined_df['fundingRateTimestamp'] == hour_timestamp)
        ]
        
        if not symbol_funding.empty and 'allocation' in positions:
            funding_rate = symbol_funding['fundingRate'].iloc[0]
            position_value = positions['allocation']
            
            if positions['linear'].side == 'short':
                funding_payment = position_value * funding_rate
            else:
                funding_payment = -position_value * funding_rate
            
            hour_funding_pnl += funding_payment
    
    return hour_funding_pnl

def calculate_unrealized_pnl(active_positions, data_manager, hour_timestamp, missing_data_log):
    """Calculate unrealized P&L for all active positions"""
    hour_unrealized_pnl = 0.0
    positions_with_data = 0
    positions_missing_data = 0
    total_nominal_exposure = 0.0
    
    for symbol, positions in active_positions.items():
        actual_spot_symbol = positions.get('actual_spot_symbol')
        scaled_allocation = positions.get('scaled_allocation', 0)
        total_nominal_exposure += scaled_allocation * 2
        
        linear_kline = data_manager.get_cached_kline(symbol, 'linear')
        spot_kline = data_manager.get_cached_kline(actual_spot_symbol, 'spot')
        
        if not linear_kline.empty and not spot_kline.empty:
            try:
                linear_current_price = get_price_at_timestamp(linear_kline, hour_timestamp)
                spot_current_price = get_price_at_timestamp(spot_kline, hour_timestamp)
                
                if linear_current_price is not None and spot_current_price is not None:
                    linear_pnl = positions['linear'].calculate_unrealized_pnl(linear_current_price)
                    spot_pnl = positions['spot'].calculate_unrealized_pnl(spot_current_price)
                    
                    hour_unrealized_pnl += linear_pnl + spot_pnl
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
                positions_missing_data += 1
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
    
    return hour_unrealized_pnl, total_nominal_exposure, positions_with_data, positions_missing_data

def update_volatility_scaling_hourly(results_df, hour_idx, vol_scaler, active_positions, 
                                     current_base_portfolio, current_scalar, logger):
    """Update volatility scaling on an hourly basis"""
    hour_timestamp = results_df.loc[hour_idx, 'timestamp']
    
    # Calculate daily return for volatility scaling (only once per day)
    if hour_idx > 0 and hour_timestamp.hour == 0:  # Daily calculation at midnight
        # Get portfolio value from 24 hours ago
        day_ago_idx = max(0, hour_idx - 24)
        prev_value = results_df.loc[day_ago_idx, 'portfolio_value']
        current_value = results_df.loc[hour_idx-1, 'portfolio_value']
        
        if prev_value > 0:
            daily_return = (current_value - prev_value) / prev_value
        else:
            daily_return = 0.0
        
        results_df.loc[hour_idx, 'daily_return'] = daily_return
        
        # Update volatility scalar
        ann_realized_vol, scalar_ewma = vol_scaler.update_volatility_scaling(daily_return)
        
        results_df.loc[hour_idx, 'realized_vol'] = ann_realized_vol
        results_df.loc[hour_idx, 'vol_scalar'] = scalar_ewma
        
        # Apply daily volatility scaling to existing positions
        if active_positions and abs(scalar_ewma - current_scalar) > 0.01:
            logger.log(f"\n🎯 DAILY VOL SCALING UPDATE at {hour_timestamp.strftime('%Y-%m-%d %H:%M')}:")
            logger.log(f"   📊 Daily return: {daily_return*100:+.2f}%")
            logger.log(f"   📈 Realized vol: {ann_realized_vol*100:.2f}% -> Target: {vol_scaler.target_vol*100:.1f}%")
            logger.log(f"   ⚡ Vol scalar: {current_scalar:.3f} -> {scalar_ewma:.3f}")
            
            # Update all active positions with new scaling
            for symbol, positions in active_positions.items():
                if symbol in current_base_portfolio:
                    base_allocation = current_base_portfolio[symbol]
                    new_scaled_allocation = base_allocation * scalar_ewma
                    old_scaled_allocation = positions.get('scaled_allocation', base_allocation)
                    
                    positions['scaled_allocation'] = new_scaled_allocation
                    positions['allocation'] = new_scaled_allocation
                    
                    scaling_change = ((new_scaled_allocation / old_scaled_allocation) - 1) * 100 if old_scaled_allocation > 0 else 0
                    logger.log(f"     🔄 {symbol}: ${old_scaled_allocation:.2f} -> ${new_scaled_allocation:.2f} ({scaling_change:+.1f}%)")
        
        return scalar_ewma
    else:
        # Copy previous values for non-daily hours
        if hour_idx > 0:
            results_df.loc[hour_idx, 'daily_return'] = results_df.loc[hour_idx-1, 'daily_return']
            results_df.loc[hour_idx, 'realized_vol'] = results_df.loc[hour_idx-1, 'realized_vol']
            results_df.loc[hour_idx, 'vol_scalar'] = results_df.loc[hour_idx-1, 'vol_scalar']
        else:
            results_df.loc[hour_idx, 'daily_return'] = 0.0
            results_df.loc[hour_idx, 'realized_vol'] = 0.10
            results_df.loc[hour_idx, 'vol_scalar'] = 1.0
        
        return current_scalar

def save_results(results_df, rebalance_history, data_availability, position_manager, 
                missing_data_log, output_dir, logger):
    """Save all backtest results to files"""
    
    # Save enhanced results
    results_path = f"{output_dir}/verbose_vol_scaled_funding_strategy_results.csv"
    results_df.to_csv(results_path, index=False)
    
    # Save data availability report
    availability_path = f"{output_dir}/verbose_vol_scaled_data_availability_report.csv"
    pd.DataFrame(data_availability).to_csv(availability_path, index=False)
    
    # Save position tracking
    position_data = []
    for pos in position_manager.position_history:
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
    
    positions_path = f"{output_dir}/verbose_vol_scaled_position_history.csv"
    pd.DataFrame(position_data).to_csv(positions_path, index=False)
    
    # Save enhanced rebalance history
    rebalance_path = f"{output_dir}/verbose_vol_scaled_rebalance_history.csv"
    pd.DataFrame(rebalance_history).to_csv(rebalance_path, index=False)
    
    # Save volatility scaling metrics
    vol_metrics = {
        'timestamp': results_df['timestamp'],
        'realized_vol': results_df['realized_vol'],
        'vol_scalar': results_df['vol_scalar'],
        'leverage': results_df['leverage'],
        'daily_return': results_df['daily_return'],
        'base_allocation_per_symbol': results_df['base_allocation_per_symbol'],
        'scaled_allocation_per_symbol': results_df['scaled_allocation_per_symbol']
    }
    vol_metrics_path = f"{output_dir}/verbose_vol_scaling_metrics.csv"
    pd.DataFrame(vol_metrics).to_csv(vol_metrics_path, index=False)
    
    # Save missing data log
    if missing_data_log:
        missing_data_df = pd.DataFrame(missing_data_log)
        missing_data_path = os.path.join(output_dir, "verbose_vol_scaled_missing_data_log.csv")
        missing_data_df.to_csv(missing_data_path, index=False)
    
    logger.log(f"\n📁 VERBOSE VOLATILITY SCALED RESULTS SAVED:")
    logger.log(f"  📊 Main results: {results_path}")
    logger.log(f"  📋 Data availability: {availability_path}")
    logger.log(f"  📈 Position history: {positions_path}")
    logger.log(f"  🔄 Rebalance history: {rebalance_path}")
    logger.log(f"  🎯 Volatility metrics: {vol_metrics_path}")
    logger.log(f"  📝 Verbose log: {logger.debug_log_path}")
    
    if missing_data_log:
        logger.log(f"  ⚠️  Missing data log: {missing_data_path}")

def backtest_funding_strategy_with_trading_verbose_vol_scaling(
    funding_data, symbols_df=None, output_dir="funding_analysis",
    start_date=None, end_date=None, 
    initial_capital=1000, min_annual_rate=30.0,
    top_n=5, rebalance_days=7,
    # Volatility scaling parameters
    target_vol=0.15, ewma_lambda=0.06,
    # Trading cost parameters
    taker_fee_pct=0.05, slippage_pct=0.03, bid_ask_spread_pct=0.02,
    # Data directories
    kline_dir="kline_data", mapping_path="future_to_spot_mapping_debug.csv",
    # Auto-download parameters
    auto_download=True, download_batch_size=10
):
    """
    REFACTORED Enhanced backtest with detailed logging AND VOLATILITY SCALING
    
    This function has been modularized into smaller, manageable components while
    maintaining the exact same functionality and output as the original.
    """
    
    # Setup and validation
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    logger = BacktestLogger(output_dir)
    
    # Set default dates if not provided
    if start_date is None:
        start_date = "2023-01-01"
        logger.log("⚠️  No start_date provided, using default: 2023-01-01")
    
    if end_date is None:
        end_date = "2025-05-16"
        logger.log("⚠️  No end_date provided, using default: 2025-05-16")
    
    # Log configuration
    logger.log(f"\n{'='*80}")
    logger.log(f"VOLATILITY SCALED BACKTEST CONFIGURATION")
    logger.log(f"{'='*80}")
    logger.log(f"Period: {start_date} to {end_date}")
    logger.log(f"Initial Capital: ${initial_capital:,.2f}")
    logger.log(f"🎯 Target Volatility: {target_vol*100:.1f}% (annualized)")
    logger.log(f"⚡ EWMA Lambda: {ewma_lambda:.3f}")
    logger.log(f"Min Annual Funding Rate: {min_annual_rate}%")
    logger.log(f"Top N Symbols: {top_n}")
    logger.log(f"Rebalance Every: {rebalance_days} days")
    
    # Validate dependencies
    missing_functions = validate_dependencies()
    if missing_functions:
        error_msg = f"❌ Missing required functions: {missing_functions}"
        logger.log(error_msg)
        logger.log("   These functions need to be imported or defined:")
        for func in missing_functions:
            logger.log(f"   - {func}")
        return None, None, None
    
    # Load symbol mapping
    try:
        symbol_mapping = load_symbol_mapping(mapping_path)
        logger.log(f"✅ Loaded mapping for {len(symbol_mapping)} symbols")
    except Exception as e:
        logger.log(f"❌ Error loading symbol mapping from {mapping_path}: {e}")
        return None, None, None
    
    # Initialize components
    vol_scaler = VolatilityScaler(target_vol, ewma_lambda)
    position_manager = PositionManager(symbol_mapping, taker_fee_pct, slippage_pct, bid_ask_spread_pct)
    data_manager = DataManager(kline_dir, auto_download)
    
    # Process dates and data
    start_dt = pd.to_datetime(start_date).tz_localize(pytz.UTC)
    end_dt = pd.to_datetime(end_date).tz_localize(pytz.UTC)
    
    combined_df = process_funding_data(funding_data, start_dt, end_dt)
    if combined_df is None:
        logger.log("❌ No funding data found for the specified period")
        return None, None, None
    
    logger.log(f"✅ Processed funding data for {len(set(combined_df['symbol']))} symbols")
    
    # Initialize results dataframe
    hours_range = pd.date_range(start=start_dt, end=end_dt, freq='H')
    results_df = pd.DataFrame(index=hours_range)
    results_df.index.name = 'timestamp'
    results_df = results_df.reset_index()
    
    # Initialize tracking columns
    columns_to_init = [
        'portfolio_value', 'funding_pnl', 'trading_pnl_realized', 'trading_pnl_unrealized',
        'trading_pnl_total', 'trading_costs', 'period_return', 'cumulative_return',
        'num_active_positions', 'daily_return', 'realized_vol', 'vol_scalar', 'leverage',
        'nominal_exposure', 'base_allocation_per_symbol', 'scaled_allocation_per_symbol'
    ]
    
    for col in columns_to_init:
        results_df[col] = 0.0
    
    results_df.loc[0, 'portfolio_value'] = initial_capital
    
    # Initialize tracking variables
    data_availability = []
    missing_data_log = []
    rebalance_history = []
    current_capital = initial_capital
    current_base_portfolio = {}
    
    # Track cumulative values
    cumulative_funding_pnl = 0.0
    cumulative_realized_pnl = 0.0
    cumulative_trading_costs = 0.0
    
    # Setup rebalance dates
    rebalance_dates = pd.date_range(start=start_dt, end=end_dt, freq=f'{rebalance_days}D')
    
    # Remove the last rebalance date if it's exactly at end_dt
    if len(rebalance_dates) > 1 and rebalance_dates[-1] >= end_dt - timedelta(hours=1):
        rebalance_dates = rebalance_dates[:-1]
        logger.log(f"⚠️  Removed last rebalance date to prevent end-of-period crash")
    
    logger.log(f"\n{'='*80}")
    logger.log(f"STARTING VOLATILITY SCALED BACKTEST - {len(rebalance_dates)} REBALANCE PERIODS")
    logger.log(f"{'='*80}")
    
    # MAIN BACKTEST LOOP
    current_scalar = 1.0
    
    for i, rebalance_date in enumerate(rebalance_dates):
        # Determine period end
        if i < len(rebalance_dates) - 1:
            next_rebalance = rebalance_dates[i+1]
        else:
            next_rebalance = end_dt
        
        period_start = rebalance_date.strftime('%Y-%m-%d %H:%M')
        period_end = next_rebalance.strftime('%Y-%m-%d %H:%M')
        
        logger.log(f"\n{'='*80}")
        logger.log(f"REBALANCE PERIOD {i+1}/{len(rebalance_dates)}")
        logger.log(f"{'='*80}")
        logger.log(f"📅 Period: {period_start} → {period_end}")
        logger.log(f"💰 Starting Capital: ${current_capital:,.2f}")
        logger.log(f"📊 Current Portfolio Positions: {len(position_manager.active_positions)}")
        
        # Get current volatility metrics
        vol_metrics = vol_scaler.get_current_metrics()
        current_scalar = vol_metrics['vol_scalar']
        
        logger.log(f"\n🎯 VOLATILITY SCALING STATUS:")
        logger.log(f"   📊 Current realized volatility: {vol_metrics['realized_vol']*100:.2f}%")
        logger.log(f"   🎯 Target volatility: {target_vol*100:.1f}%")
        logger.log(f"   ⚡ Current vol scalar: {current_scalar:.3f}")
        
        # Select symbols for trading
        top_symbols = select_trading_symbols(combined_df, rebalance_date, rebalance_days, 
                                           min_annual_rate, top_n)
        
        # Track the rebalance trading cost for this period
        period_trading_cost = 0.0
        period_realized_pnl = 0.0
        
        if not top_symbols.empty:
            logger.log(f"\n⭐ TOP {min(top_n, len(top_symbols))} SYMBOLS BY FUNDING RATE:")
            logger.log(f"{'Rank':<5} {'Symbol':<15} {'Funding Rate':<15} {'Interval':<10}")
            logger.log(f"{'-'*50}")
            for idx, (_, row) in enumerate(top_symbols.iterrows()):
                interval_hours = int(row['funding_interval_min'] / 60)
                logger.log(f"{idx+1:<5} {row['symbol']:<15} {row['annualized_rate']:<15.2f}% {interval_hours}h")
            
            # Calculate allocations
            base_allocation_per_symbol = current_capital / min(top_n, len(top_symbols))
            scaled_allocation_per_symbol = base_allocation_per_symbol * current_scalar
            
            logger.log(f"\n💰 ALLOCATION CALCULATION:")
            logger.log(f"   💵 Available capital: ${current_capital:,.2f}")
            logger.log(f"   📊 Base allocation per symbol: ${base_allocation_per_symbol:.2f}")
            logger.log(f"   ⚡ Vol scalar: {current_scalar:.3f}")
            logger.log(f"   💎 Scaled allocation per symbol: ${scaled_allocation_per_symbol:.2f}")
            
            # Handle position changes
            symbols_to_remove = set(position_manager.active_positions.keys()) - set(top_symbols['symbol'])
            symbols_to_update = set(position_manager.active_positions.keys()) & set(top_symbols['symbol'])
            new_symbols = set(top_symbols['symbol']) - set(position_manager.active_positions.keys())
            
            # Close positions that are no longer in portfolio
            if symbols_to_remove:
                logger.log(f"\n❌ CLOSING POSITIONS (no longer in top {top_n}):")
                for symbol in symbols_to_remove:
                    try:
                        positions = position_manager.active_positions.get(symbol, {})
                        actual_spot_symbol = positions.get('actual_spot_symbol', symbol_mapping.get(symbol, symbol.replace('USDT', '')))
                        
                        linear_kline = data_manager.get_cached_kline(symbol, 'linear')
                        spot_kline = data_manager.get_cached_kline(actual_spot_symbol, 'spot')
                        
                        if not linear_kline.empty and not spot_kline.empty:
                            linear_exit_price = get_price_at_timestamp(linear_kline, rebalance_date)
                            spot_exit_price = get_price_at_timestamp(spot_kline, rebalance_date)
                            
                            if linear_exit_price is not None and spot_exit_price is not None:
                                total_pnl, cost, realized_pnl = position_manager.close_position(
                                    symbol, linear_exit_price, spot_exit_price, rebalance_date)
                                
                                period_trading_cost += cost
                                period_realized_pnl += realized_pnl
                                
                                logger.log(f"  ✅ Closed {symbol}, PnL: ${total_pnl:.2f}, Cost: ${cost:.2f}")
                    except Exception as e:
                        logger.log(f"  ⚠️  Error closing {symbol}: {e}")
            
            # Update existing positions with new scaled allocation
            if symbols_to_update:
                logger.log(f"\n🔄 UPDATING EXISTING POSITIONS WITH NEW VOL SCALING:")
                position_manager.update_position_allocations(
                    {symbol: base_allocation_per_symbol for symbol in top_symbols['symbol']}, 
                    current_scalar)
                
                for symbol in symbols_to_update:
                    if symbol in position_manager.active_positions:
                        old_allocation = position_manager.active_positions[symbol].get('allocation', 0)
                        allocation_change_pct = ((scaled_allocation_per_symbol - old_allocation) / old_allocation * 100) if old_allocation > 0 else 0
                        logger.log(f"  🔄 {symbol}: ${old_allocation:.2f} → ${scaled_allocation_per_symbol:.2f} ({allocation_change_pct:+.1f}%)")
            
            # Open new positions
            if new_symbols:
                logger.log(f"\n🔍 CHECKING DATA AVAILABILITY FOR NEW POSITIONS:")
                symbols_to_trade = []
                
                for _, row in top_symbols.iterrows():
                    symbol = row['symbol']
                    
                    if symbol in position_manager.active_positions:
                        continue
                    
                    spot_symbol = symbol_mapping.get(symbol, symbol.replace('USDT', ''))
                    logger.log(f"  🔍 Processing symbol: {symbol} -> {spot_symbol}")
                    
                    # Get launch time for this symbol
                    launch_time = None
                    if symbols_df is not None:
                        symbol_row = symbols_df[symbols_df['symbol'] == symbol]
                        if not symbol_row.empty and 'launchTime' in symbol_row.columns:
                            launch_time = symbol_row['launchTime'].iloc[0]
                    
                    # Load data
                    linear_kline, spot_kline = data_manager.load_symbol_data(
                        symbol, spot_symbol, start_date, end_date, launch_time)
                    
                    # Check data availability
                    linear_available = not linear_kline.empty
                    spot_available = not spot_kline.empty
                    
                    try:
                        linear_price = get_price_at_timestamp(linear_kline, rebalance_date) if linear_available else None
                        spot_price = get_price_at_timestamp(spot_kline, rebalance_date) if spot_available else None
                    except Exception as e:
                        logger.log(f"  ⚠️  Error getting prices for {symbol}: {e}")
                        linear_price = spot_price = None
                    
                    # Determine if we can trade
                    if linear_available and spot_available and linear_price is not None and spot_price is not None:
                        logger.log(f"  📊 {symbol} -> {spot_symbol}: Linear ✅ Spot ✅ Prices ✅")
                        symbols_to_trade.append({
                            'symbol': symbol,
                            'spot_symbol': spot_symbol,
                            'linear_price': linear_price,
                            'spot_price': spot_price,
                            'funding_rate': row['annualized_rate'],
                            'is_proxy': False,
                            'actual_spot_symbol': spot_symbol
                        })
                    elif linear_available and linear_price is not None:
                        logger.log(f"  📊 {symbol} -> {spot_symbol}: Linear ✅ Spot ❌ - TRYING BITCOIN PROXY")
                        btc_price, btc_available = data_manager.try_bitcoin_proxy(
                            symbol, rebalance_date, start_date, end_date, logger)
                        
                        if btc_available and btc_price is not None:
                            symbols_to_trade.append({
                                'symbol': symbol,
                                'spot_symbol': 'BTCUSDT',
                                'linear_price': linear_price,
                                'spot_price': btc_price,
                                'funding_rate': row['annualized_rate'],
                                'is_proxy': True,
                                'actual_spot_symbol': 'BTCUSDT'
                            })
                    else:
                        logger.log(f"  ❌ {symbol} -> {spot_symbol}: Cannot trade without linear data")
                    
                    # Record data availability
                    data_availability.append({
                        'rebalance_date': rebalance_date,
                        'symbol': symbol,
                        'spot_symbol': spot_symbol,
                        'linear_data_available': linear_available,
                        'spot_data_available': spot_available,
                        'both_available': linear_available and spot_available,
                        'linear_price_available': linear_price is not None,
                        'spot_price_available': spot_price is not None,
                        'prices_available': linear_price is not None and spot_price is not None,
                        'entry_timestamp': rebalance_date,
                        'linear_price': linear_price,
                        'spot_price': spot_price,
                        'avg_funding_rate': row['annualized_rate']
                    })
                
                # Open new positions
                if symbols_to_trade:
                    logger.log(f"\n✅ OPENING NEW POSITIONS WITH VOL SCALING:")
                    logger.log(f"Symbols to trade: {len(symbols_to_trade)}")
                    
                    for symbol_info in symbols_to_trade:
                        symbol = symbol_info['symbol']
                        spot_symbol = symbol_info['spot_symbol']
                        linear_price = symbol_info['linear_price']
                        spot_price = symbol_info['spot_price']
                        avg_funding_rate = symbol_info['funding_rate']
                        is_proxy = symbol_info.get('is_proxy', False)
                        actual_spot_symbol = symbol_info.get('actual_spot_symbol', spot_symbol)
                        
                        total_cost, linear_cost, spot_cost = position_manager.open_position(
                            symbol, actual_spot_symbol, linear_price, spot_price,
                            scaled_allocation_per_symbol, avg_funding_rate, rebalance_date,
                            base_allocation_per_symbol, current_scalar, is_proxy)
                        
                        period_trading_cost += total_cost
                        
                        proxy_indicator = " 🪙 [BTC PROXY]" if is_proxy else ""
                        scaling_indicator = f" ⚡ [SCALED {current_scalar:.2f}x]"
                        logger.log(f"\n  📈 {symbol} ({avg_funding_rate:+.2f}% funding){proxy_indicator}{scaling_indicator}:")
                        logger.log(f"     💸 Trading costs: ${total_cost:.2f}")
            
            # Update current base portfolio for tracking
            current_base_portfolio = {symbol: base_allocation_per_symbol for symbol in top_symbols['symbol']}
            
            # Calculate total nominal exposure after rebalancing
            total_nominal_exposure = sum(pos.get('scaled_allocation', 0) * 2 
                                       for pos in position_manager.active_positions.values())
            final_leverage = total_nominal_exposure / current_capital if current_capital > 0 else 0.0
            
            logger.log(f"\n📊 REBALANCE PERIOD {i+1} SUMMARY:")
            logger.log(f"✅ Opened {len(symbols_to_trade) if 'symbols_to_trade' in locals() else 0} new positions")
            logger.log(f"❌ Closed {len(symbols_to_remove)} old positions")
            logger.log(f"🔄 Updated {len(symbols_to_update)} existing positions")
            logger.log(f"💰 Total active positions: {len(position_manager.active_positions)}")
            logger.log(f"💸 Period trading costs: ${period_trading_cost:.2f}")
            logger.log(f"📈 Period realized PnL: ${period_realized_pnl:.2f}")
            logger.log(f"⚖️  Final leverage: {final_leverage:.2f}x")
            
        else:
            logger.log(f"\n❌ NO SYMBOLS MEET FUNDING RATE CRITERIA - CLOSING ALL POSITIONS")
            
            # Close all positions
            for symbol in list(position_manager.active_positions.keys()):
                try:
                    positions = position_manager.active_positions.get(symbol, {})
                    actual_spot_symbol = positions.get('actual_spot_symbol', symbol_mapping.get(symbol, symbol.replace('USDT', '')))
                    
                    linear_kline = data_manager.get_cached_kline(symbol, 'linear')
                    spot_kline = data_manager.get_cached_kline(actual_spot_symbol, 'spot')
                    
                    if not linear_kline.empty and not spot_kline.empty:
                        linear_exit_price = get_price_at_timestamp(linear_kline, rebalance_date)
                        spot_exit_price = get_price_at_timestamp(spot_kline, rebalance_date)
                        
                        if linear_exit_price is not None and spot_exit_price is not None:
                            total_pnl, cost, realized_pnl = position_manager.close_position(
                                symbol, linear_exit_price, spot_exit_price, rebalance_date)
                            period_realized_pnl += realized_pnl
                            logger.log(f"  ✅ Closed all positions for {symbol}, PnL: ${total_pnl:.2f}")
                except Exception as e:
                    logger.log(f"  ❌ Error closing {symbol}: {e}")
            
            current_base_portfolio = {}
        
        # Update cumulative values
        cumulative_trading_costs += period_trading_cost
        cumulative_realized_pnl += period_realized_pnl
        
        # Record rebalance event
        rebalance_event = {
            'date': rebalance_date,
            'period': f"{i+1}/{len(rebalance_dates)}",
            'starting_capital': current_capital,
            'symbols_selected': len(symbols_to_trade) if 'symbols_to_trade' in locals() else 0,
            'trading_costs': period_trading_cost,
            'realized_pnl': period_realized_pnl,
            'portfolio_value': current_capital,
            'active_positions': len(position_manager.active_positions),
            'base_allocation_per_symbol': base_allocation_per_symbol if not top_symbols.empty else 0,
            'scaled_allocation_per_symbol': scaled_allocation_per_symbol if not top_symbols.empty else 0,
            'vol_scalar': current_scalar,
            'realized_vol': vol_metrics['realized_vol'],
            'target_vol': target_vol,
            'cumulative_funding_pnl': cumulative_funding_pnl,
            'cumulative_realized_pnl': cumulative_realized_pnl,
            'cumulative_trading_costs': cumulative_trading_costs
        }
        rebalance_history.append(rebalance_event)
        
        # Process hourly data for this period
        period_hours = results_df[
            (results_df['timestamp'] >= rebalance_date) & 
            (results_df['timestamp'] < next_rebalance)
        ].index.tolist()
        
        rebalance_cost_recorded = False
        period_funding_pnl = 0.0
        
        for hour_idx in period_hours:
            hour_timestamp = results_df.loc[hour_idx, 'timestamp']
            
            # Update volatility scaling
            current_scalar = update_volatility_scaling_hourly(
                results_df, hour_idx, vol_scaler, position_manager.active_positions,
                current_base_portfolio, current_scalar, logger)
            
            # Record trading costs and realized PnL at the rebalance hour
            if hour_timestamp >= rebalance_date and not rebalance_cost_recorded:
                if period_trading_cost > 0:
                    results_df.loc[hour_idx, 'trading_costs'] = -period_trading_cost
                if period_realized_pnl != 0:
                    results_df.loc[hour_idx, 'trading_pnl_realized'] = period_realized_pnl
                rebalance_cost_recorded = True
            
            # Calculate funding PnL
            hour_funding_pnl = calculate_funding_pnl(position_manager.active_positions, combined_df, hour_timestamp)
            period_funding_pnl += hour_funding_pnl
            cumulative_funding_pnl += hour_funding_pnl
            
            # Calculate trading PnL (unrealized)
            hour_unrealized_pnl, total_nominal_exposure, positions_with_data, positions_missing_data = calculate_unrealized_pnl(
                position_manager.active_positions, data_manager, hour_timestamp, missing_data_log)
            
            # Calculate portfolio value and other metrics
            portfolio_value = initial_capital + cumulative_funding_pnl + hour_unrealized_pnl + cumulative_realized_pnl - cumulative_trading_costs
            leverage = total_nominal_exposure / portfolio_value if portfolio_value > 0 else 0.0
            
            # Update results
            results_df.loc[hour_idx, 'funding_pnl'] = hour_funding_pnl
            results_df.loc[hour_idx, 'trading_pnl_unrealized'] = hour_unrealized_pnl
            results_df.loc[hour_idx, 'trading_pnl_total'] = hour_unrealized_pnl + cumulative_realized_pnl
            results_df.loc[hour_idx, 'num_active_positions'] = len(position_manager.active_positions)
            results_df.loc[hour_idx, 'portfolio_value'] = portfolio_value
            results_df.loc[hour_idx, 'leverage'] = leverage
            results_df.loc[hour_idx, 'nominal_exposure'] = total_nominal_exposure
            
            # Track allocations
            if current_base_portfolio:
                base_total = sum(current_base_portfolio.values())
                scaled_total = base_total * results_df.loc[hour_idx, 'vol_scalar']
                results_df.loc[hour_idx, 'base_allocation_per_symbol'] = base_total / len(current_base_portfolio)
                results_df.loc[hour_idx, 'scaled_allocation_per_symbol'] = scaled_total / len(current_base_portfolio)
        
        # Update current capital for next rebalance
        if len(period_hours) > 0 and i < len(rebalance_dates) - 1:
            last_hour_idx = period_hours[-1] 
            current_capital = results_df.loc[last_hour_idx, 'portfolio_value']
        
        # Period summary
        portfolio_change = current_capital - rebalance_event.get('starting_capital', initial_capital)
        logger.log(f"\n📈 END OF PERIOD {i+1} SUMMARY:")
        logger.log(f"🏦 Portfolio Value: ${rebalance_event.get('starting_capital', initial_capital):,.2f} → ${current_capital:,.2f} ({portfolio_change:+.2f})")
        logger.log(f"💰 Period Funding PnL: ${period_funding_pnl:.2f}")
        logger.log(f"📊 Period Realized Trading PnL: ${period_realized_pnl:.2f}")
        logger.log(f"💸 Period Trading Costs: ${period_trading_cost:.2f}")
    
    # FINAL CLOSE ALL POSITIONS
    logger.log(f"\n{'='*80}")
    logger.log(f"CLOSING ALL REMAINING POSITIONS AT END OF BACKTEST")
    logger.log(f"{'='*80}")
    
    final_realized_pnl = 0.0
    final_trading_costs = 0.0
    
    if position_manager.active_positions:
        logger.log(f"Closing {len(position_manager.active_positions)} remaining positions")
        
        for symbol in list(position_manager.active_positions.keys()):
            try:
                positions = position_manager.active_positions.get(symbol, {})
                actual_spot_symbol = positions.get('actual_spot_symbol', symbol_mapping.get(symbol, symbol.replace('USDT', '')))
                
                linear_kline = data_manager.get_cached_kline(symbol, 'linear')
                spot_kline = data_manager.get_cached_kline(actual_spot_symbol, 'spot')
                
                if not linear_kline.empty and not spot_kline.empty:
                    close_timestamp = end_dt - timedelta(hours=1)
                    linear_exit_price = get_price_at_timestamp(linear_kline, close_timestamp)
                    spot_exit_price = get_price_at_timestamp(spot_kline, close_timestamp)
                    
                    if linear_exit_price is not None and spot_exit_price is not None:
                        total_pnl, cost, realized_pnl = position_manager.close_position(
                            symbol, linear_exit_price, spot_exit_price, close_timestamp)
                        
                        final_realized_pnl += realized_pnl
                        final_trading_costs += cost
                        logger.log(f"  ✅ Final close {symbol}: PnL ${total_pnl:.2f}, Cost: ${cost:.2f}")
            except Exception as e:
                logger.log(f"  ⚠️  Error closing final positions for {symbol}: {e}")
        
        # Update cumulative values with final closes
        cumulative_realized_pnl += final_realized_pnl
        cumulative_trading_costs += final_trading_costs
        
        # Record final costs in the last row
        last_idx = len(results_df) - 1
        if final_trading_costs > 0:
            results_df.loc[last_idx, 'trading_costs'] += -final_trading_costs
        if final_realized_pnl != 0:
            results_df.loc[last_idx, 'trading_pnl_realized'] += final_realized_pnl
    
    # RECALCULATE FINAL PORTFOLIO VALUES
    final_portfolio_value = initial_capital + cumulative_funding_pnl + cumulative_realized_pnl - cumulative_trading_costs
    
    # Set final portfolio value (no unrealized PnL since all positions are closed)
    results_df.loc[len(results_df)-1, 'portfolio_value'] = final_portfolio_value
    results_df.loc[len(results_df)-1, 'trading_pnl_unrealized'] = 0.0
    results_df.loc[len(results_df)-1, 'trading_pnl_total'] = cumulative_realized_pnl
    results_df.loc[len(results_df)-1, 'num_active_positions'] = 0
    results_df.loc[len(results_df)-1, 'leverage'] = 0.0
    results_df.loc[len(results_df)-1, 'nominal_exposure'] = 0.0
    
    logger.log(f"✅ Final portfolio value: ${final_portfolio_value:.2f}")
    
    # Calculate final performance metrics
    results_df['period_return'] = results_df['portfolio_value'].pct_change()
    results_df.loc[0, 'period_return'] = (results_df.loc[0, 'portfolio_value'] - initial_capital) / initial_capital
    results_df['period_return'] = results_df['period_return'].fillna(0).replace([np.inf, -np.inf], 0)
    results_df['cumulative_return'] = (results_df['portfolio_value'] / initial_capital) - 1
    
    # Add cumulative columns for plotting
    results_df['cum_funding_pnl'] = results_df['funding_pnl'].cumsum()
    results_df['cum_trading_pnl'] = results_df['trading_pnl_total']
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
    
    # Calculate volatility metrics
    final_realized_vol = results_df['realized_vol'].iloc[-1]
    avg_vol_scalar = results_df['vol_scalar'].mean()
    max_vol_scalar = results_df['vol_scalar'].max()
    min_vol_scalar = results_df['vol_scalar'].min()
    avg_leverage = results_df['leverage'].mean()
    max_leverage = results_df['leverage'].max()
    
    # COMPREHENSIVE FINAL SUMMARY
    logger.log(f"\n{'='*80}")
    logger.log(f"FINAL VOLATILITY SCALED BACKTEST RESULTS")
    logger.log(f"{'='*80}")
    logger.log(f"🏁 STRATEGY PERFORMANCE:")
    logger.log(f"   📅 Period: {start_date} to {end_date} ({total_days} days)")
    logger.log(f"   💰 Initial Capital: ${initial_capital:,.2f}")
    logger.log(f"   🏦 Final Portfolio Value: ${results_df['portfolio_value'].iloc[-1]:,.2f}")
    logger.log(f"   📈 Total Return: {total_return:.2%}")
    logger.log(f"   📊 Annualized Return: {annual_return:.2%}")
    logger.log(f"   📉 Maximum Drawdown: {max_drawdown:.2%}")
    logger.log(f"   ⚡ Sharpe Ratio: {sharpe_ratio:.2f}")
    logger.log(f"   📊 Annualized Volatility: {volatility:.2%}")
    
    logger.log(f"\n🎯 VOLATILITY SCALING PERFORMANCE:")
    logger.log(f"   🎯 Target Volatility: {target_vol*100:.1f}%")
    logger.log(f"   📊 Final Realized Volatility: {final_realized_vol*100:.2f}%")
    logger.log(f"   ⚡ Average Vol Scalar: {avg_vol_scalar:.3f}")
    logger.log(f"   📈 Max Vol Scalar: {max_vol_scalar:.3f}")
    logger.log(f"   📉 Min Vol Scalar: {min_vol_scalar:.3f}")
    logger.log(f"   ⚖️  Average Leverage: {avg_leverage:.2f}x")
    logger.log(f"   ⚖️  Maximum Leverage: {max_leverage:.2f}x")
    
    # Calculate total costs and PnL
    total_trading_costs = abs(results_df['cum_trading_costs'].iloc[-1])
    total_funding_pnl = results_df['cum_funding_pnl'].iloc[-1]
    total_trading_pnl = results_df['cum_trading_pnl'].iloc[-1]
    total_realized_pnl = sum(pos.realized_pnl for pos in position_manager.position_history if pos.is_closed)
    
    logger.log(f"\n💰 P&L BREAKDOWN:")
    logger.log(f"   💸 Total Trading Costs: ${total_trading_costs:.2f}")
    logger.log(f"   💰 Total Funding PnL: ${total_funding_pnl:.2f}")
    logger.log(f"   📊 Total Trading PnL: ${total_trading_pnl:.2f}")
    logger.log(f"   💎 Total Realized Trading PnL: ${total_realized_pnl:.2f}")
    
    pnl_net = total_funding_pnl + total_trading_pnl - total_trading_costs
    logger.log(f"   🏆 Net P&L: ${pnl_net:.2f}")
    
    # Enhanced cost analysis
    cost_pct = (total_trading_costs / initial_capital) * 100
    funding_pct = (total_funding_pnl / initial_capital) * 100
    vol_scaling_benefit = (avg_vol_scalar - 1) * 100
    
    logger.log(f"\n📊 ENHANCED COST ANALYSIS:")
    logger.log(f"   💸 Trading costs as % of capital: {cost_pct:.2f}%")
    logger.log(f"   💰 Funding PnL as % of capital: {funding_pct:.2f}%")
    logger.log(f"   ⚖️  Funding vs Costs ratio: {abs(total_funding_pnl/total_trading_costs):.2f}x" if total_trading_costs > 0 else "   ⚖️  Funding vs Costs ratio: N/A")
    logger.log(f"   ⚡ Vol scaling avg benefit: {vol_scaling_benefit:+.1f}%")
    
    # Data availability summary
    if data_availability:
        total_evaluations = len(data_availability)
        successful_trades = len([d for d in data_availability if d['both_available']])
        success_rate = (successful_trades / total_evaluations) * 100
        
        logger.log(f"\n📊 DATA AVAILABILITY:")
        logger.log(f"   📋 Total symbol evaluations: {total_evaluations}")
        logger.log(f"   ✅ Successful data availability: {successful_trades}")
        logger.log(f"   📈 Success rate: {success_rate:.1f}%")
    
    # Trading frequency
    total_positions = len(position_manager.position_history)
    avg_positions_per_rebalance = total_positions / len(rebalance_dates) if len(rebalance_dates) > 0 else 0
    
    logger.log(f"\n📊 TRADING ACTIVITY:")
    logger.log(f"   🔄 Total rebalance periods: {len(rebalance_dates)}")
    logger.log(f"   📈 Total positions opened: {total_positions}")
    logger.log(f"   📊 Average positions per rebalance: {avg_positions_per_rebalance:.1f}")
    
    # Volatility scaling effectiveness summary
    vol_target_deviation = abs(final_realized_vol - target_vol) / target_vol * 100
    vol_adherence_periods = len([x for x in results_df['realized_vol'] if abs(x - target_vol) / target_vol < 0.2])
    vol_adherence_rate = vol_adherence_periods / len(results_df) * 100
    
    logger.log(f"\n🎯 VOLATILITY SCALING EFFECTIVENESS:")
    logger.log(f"   📊 Target Deviation: {vol_target_deviation:.1f}%")
    logger.log(f"   📊 Periods within 20% of target vol: {vol_adherence_periods}/{len(results_df)} ({vol_adherence_rate:.1f}%)")
    logger.log(f"   ⚡ EWMA Lambda effectiveness: {ewma_lambda:.3f}")
    logger.log(f"   📈 Vol scaling helped achieve target: {'YES' if vol_target_deviation < 50 else 'PARTIALLY' if vol_target_deviation < 100 else 'NO'}")
    
    logger.log(f"\n{'='*80}")
    logger.log(f"VOLATILITY SCALED BACKTEST COMPLETED SUCCESSFULLY")
    logger.log(f"{'='*80}")
    
    # Save all results
    save_results(results_df, rebalance_history, data_availability, position_manager, 
                missing_data_log, output_dir, logger)
    
    print(f"\n📁 VERBOSE VOLATILITY SCALED RESULTS SAVED:")
    print(f"  📊 Main results: {output_dir}/verbose_vol_scaled_funding_strategy_results.csv")
    print(f"  📋 Data availability: {output_dir}/verbose_vol_scaled_data_availability_report.csv")
    print(f"  📈 Position history: {output_dir}/verbose_vol_scaled_position_history.csv")
    print(f"  🔄 Rebalance history: {output_dir}/verbose_vol_scaled_rebalance_history.csv")
    print(f"  🎯 Volatility metrics: {output_dir}/verbose_vol_scaling_metrics.csv")
    print(f"  📝 Verbose log: {logger.debug_log_path}")
    
    if missing_data_log:
        print(f"  ⚠️  Missing data log: {output_dir}/verbose_vol_scaled_missing_data_log.csv")
    
    # Plot results using enhanced plotting function for volatility scaling
    try:
        plot_enhanced_vol_scaled_backtest_results(results_df[:-1], rebalance_history, output_dir, initial_capital, target_vol)
    except NameError:
        logger.log("⚠️  plot_enhanced_vol_scaled_backtest_results function not found, skipping plotting")
        print("⚠️  plot_enhanced_vol_scaled_backtest_results function not found, skipping plotting")
    
    return results_df, rebalance_history, data_availability



def plot_enhanced_vol_scaled_backtest_results(results_df, rebalance_history, output_dir, initial_capital, target_vol):
    """
    Enhanced plotting function for volatility scaled backtest results
    """
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    fig.suptitle('Enhanced Volatility Scaled Funding Strategy Performance', fontsize=16, fontweight='bold')
    
    # Plot 1: Portfolio Value
    ax1 = axes[0, 0]
    ax1.plot(results_df['timestamp'], results_df['portfolio_value'], 'b-', linewidth=1.5)
    ax1.axhline(y=initial_capital, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
    
    # Add rebalance markers
    for event in rebalance_history:
        ax1.axvline(x=event['date'], color='g', linestyle=':', alpha=0.3)
    
    ax1.set_title('Portfolio Value Over Time', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Portfolio Value (USD)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Cumulative Return
    ax2 = axes[0, 1]
    cumulative_return = (results_df['portfolio_value'] / initial_capital - 1) * 100
    ax2.plot(results_df['timestamp'], cumulative_return, 'g-', linewidth=1.5)
    ax2.set_title('Cumulative Return (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Return (%)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Volatility Tracking
    ax3 = axes[0, 2]
    ax3.plot(results_df['timestamp'], results_df['realized_vol'] * 100, 'orange', linewidth=1.5, label='Realized Vol')
    ax3.axhline(y=target_vol * 100, color='red', linestyle='--', linewidth=2, label=f'Target Vol ({target_vol*100:.1f}%)')
    ax3.set_title('Volatility Tracking', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Annualized Volatility (%)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Volatility Scalar
    ax4 = axes[1, 0]
    ax4.plot(results_df['timestamp'], results_df['vol_scalar'], 'purple', linewidth=1.5)
    ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline (1.0)')
    ax4.set_title('Volatility Scalar', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Scalar Value')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Plot 5: Leverage
    ax5 = axes[1, 1]
    ax5.plot(results_df['timestamp'], results_df['leverage'], 'red', linewidth=1.5)
    ax5.set_title('Leverage Over Time', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Leverage (x)')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: P&L Components
    ax6 = axes[1, 2]
    ax6.plot(results_df['timestamp'], results_df['cum_funding_pnl'], 'green', linewidth=1.5, label='Funding PnL')
    ax6.plot(results_df['timestamp'], results_df['cum_trading_pnl'], 'blue', linewidth=1.5, label='Trading PnL')
    ax6.plot(results_df['timestamp'], results_df['cum_trading_costs'], 'red', linewidth=1.5, label='Trading Costs')
    ax6.set_title('P&L Components', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Cumulative P&L (USD)')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    # Plot 7: Allocation Scaling
    ax7 = axes[2, 0]
    ax7.plot(results_df['timestamp'], results_df['base_allocation_per_symbol'], 'blue', linewidth=1.5, label='Base Allocation', alpha=0.7)
    ax7.plot(results_df['timestamp'], results_df['scaled_allocation_per_symbol'], 'red', linewidth=1.5, label='Scaled Allocation')
    ax7.set_title('Allocation Scaling', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Allocation per Symbol (USD)')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    # Plot 8: Drawdowns
    ax8 = axes[2, 1]
    portfolio_peak = results_df['portfolio_value'].cummax()
    drawdown = (results_df['portfolio_value'] - portfolio_peak) / portfolio_peak * 100
    ax8.fill_between(results_df['timestamp'], drawdown, 0, color='red', alpha=0.3)
    ax8.set_title('Drawdowns (%)', fontsize=12, fontweight='bold')
    ax8.set_ylabel('Drawdown (%)')
    ax8.grid(True, alpha=0.3)
    ax8.invert_yaxis()
    
    # Plot 9: Daily Returns Distribution
    ax9 = axes[2, 2]
    daily_returns = results_df['daily_return'].dropna() * 100
    if len(daily_returns) > 0:
        ax9.hist(daily_returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax9.axvline(daily_returns.mean(), color='red', linestyle='--', label=f'Mean: {daily_returns.mean():.3f}%')
        ax9.axvline(0, color='gray', linestyle='-', alpha=0.5)
    ax9.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
    ax9.set_xlabel('Daily Return (%)')
    ax9.set_ylabel('Frequency')
    ax9.grid(True, alpha=0.3)
    ax9.legend()
    
    # Format x-axis for all time series plots
    for i in range(3):
        for j in range(3):
            if i < 2 or j < 2:  # Skip the histogram
                axes[i, j].tick_params(axis='x', rotation=45)
    
    # Calculate and display comprehensive metrics
    total_return = (results_df['portfolio_value'].iloc[-1] / initial_capital - 1) * 100
    final_vol = results_df['realized_vol'].iloc[-1] * 100
    avg_leverage = results_df['leverage'].mean()
    max_drawdown = drawdown.min()
    
    metrics_text = (
        f"PERFORMANCE METRICS:\n"
        f"Total Return: {total_return:.2f}%\n"
        f"Target Volatility: {target_vol*100:.1f}%\n"
        f"Final Realized Vol: {final_vol:.2f}%\n"
        f"Avg Leverage: {avg_leverage:.2f}x\n"
        f"Max Drawdown: {max_drawdown:.2f}%\n"
        f"Final Value: ${results_df['portfolio_value'].iloc[-1]:,.0f}"
    )
    
    fig.text(0.02, 0.02, metrics_text, fontsize=10, fontweight='bold',
             bbox=dict(facecolor='lightblue', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    
    # Save figure
    output_path = f"{output_dir}/enhanced_vol_scaled_funding_strategy_performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Enhanced performance chart saved to {output_path}")
    
    plt.show()




# Import Bybit kline functions for downloading missing data
try:
    from utility.bybit_klines import parallel_download_data, download_kline_data
    from pybit.unified_trading import HTTP
    BYBIT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Bybit kline functions not available: {e}")
    BYBIT_AVAILABLE = False

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
    
def analyze_data_availability(data_availability, output_dir="funding_analysis"):
    """
    Analyze and visualize data availability for the backtest
    
    Args:
        data_availability (list): List of data availability records
        output_dir (str): Directory to save results
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    if not data_availability:
        print("❌ No data availability records to analyze")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data_availability)
    
    print(f"\n📊 DATA AVAILABILITY ANALYSIS")
    print(f"{'='*50}")
    print(f"Total symbol evaluations: {len(df)}")
    print(f"Unique symbols: {df['symbol'].nunique()}")
    print(f"Date range: {df['rebalance_date'].min()} to {df['rebalance_date'].max()}")
    
    # Add missing columns if they don't exist
    required_columns = ['bitcoin_proxy_used', 'bitcoin_proxy_attempted']
    for col in required_columns:
        if col not in df.columns:
            df[col] = False
    
    # Overall success rates
    print(f"\n🎯 SUCCESS RATES:")
    print(f"Linear data available: {df['linear_data_available'].mean():.1%}")
    print(f"Spot data available: {df['spot_data_available'].mean():.1%}")
    print(f"Both available (original): {df['both_available'].mean():.1%}")
    print(f"Prices available: {df['prices_available'].mean():.1%}")
    
    if df['bitcoin_proxy_attempted'].any():
        proxy_attempted = df['bitcoin_proxy_attempted'].sum()
        proxy_successful = df['bitcoin_proxy_used'].sum()
        print(f"Bitcoin proxy attempted: {proxy_attempted} times")
        print(f"Bitcoin proxy successful: {proxy_successful} times ({proxy_successful/proxy_attempted:.1%})")
        
        # Recalculate success rate including proxy
        df['final_success'] = df['both_available'] | df['bitcoin_proxy_used']
        print(f"Final success rate (including proxy): {df['final_success'].mean():.1%}")
    
    # Symbol-by-symbol analysis  
    agg_dict = {
        'both_available': 'mean',
        'linear_data_available': 'mean',
        'spot_data_available': 'mean',
        'prices_available': 'mean'
    }
    
    # Only add bitcoin proxy columns if they exist in the dataframe
    if 'bitcoin_proxy_used' in df.columns:
        agg_dict['bitcoin_proxy_used'] = 'any'
    if 'bitcoin_proxy_attempted' in df.columns:
        agg_dict['bitcoin_proxy_attempted'] = 'any'
    
    symbol_availability = df.groupby('symbol').agg(agg_dict).reset_index()
    
    symbol_availability = symbol_availability.sort_values('both_available', ascending=False)
    
    print(f"\n📋 TOP 10 SYMBOLS BY DATA AVAILABILITY:")
    print(f"{'Symbol':<15} {'Linear':<8} {'Spot':<6} {'Both':<6} {'Proxy':<6}")
    print(f"{'-'*50}")
    
    for _, row in symbol_availability.head(10).iterrows():
        proxy_status = "✅" if row.get('bitcoin_proxy_used', False) else ("🔄" if row.get('bitcoin_proxy_attempted', False) else "❌")
        print(f"{row['symbol']:<15} {row['linear_data_available']:<8.0%} {row['spot_data_available']:<6.0%} {row['both_available']:<6.0%} {proxy_status:<6}")
    
    print(f"\n📋 BOTTOM 10 SYMBOLS BY DATA AVAILABILITY:")
    print(f"{'Symbol':<15} {'Linear':<8} {'Spot':<6} {'Both':<6} {'Proxy':<6}")
    print(f"{'-'*50}")
    
    for _, row in symbol_availability.tail(10).iterrows():
        proxy_status = "✅" if row.get('bitcoin_proxy_used', False) else ("🔄" if row.get('bitcoin_proxy_attempted', False) else "❌")
        print(f"{row['symbol']:<15} {row['linear_data_available']:<8.0%} {row['spot_data_available']:<6.0%} {row['both_available']:<6.0%} {proxy_status:<6}")
    
    # Time-based analysis
    if 'rebalance_date' in df.columns:
        time_availability = df.groupby('rebalance_date').agg({
            'both_available': 'mean',
            'linear_data_available': 'mean', 
            'spot_data_available': 'mean',
            'bitcoin_proxy_used': 'sum',
            'symbol': 'count'
        }).reset_index()
        
        print(f"\n📅 DATA AVAILABILITY OVER TIME:")
        print(f"{'Date':<12} {'Symbols':<8} {'Linear':<8} {'Spot':<6} {'Both':<6} {'Proxy':<6}")
        print(f"{'-'*55}")
        
        for _, row in time_availability.iterrows():
            date_str = row['rebalance_date'].strftime('%Y-%m-%d')
            print(f"{date_str:<12} {row['symbol']:<8.0f} {row['linear_data_available']:<8.0%} {row['spot_data_available']:<6.0%} {row['both_available']:<6.0%} {row['bitcoin_proxy_used']:<6.0f}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Overall availability rates
    ax1 = axes[0, 0]
    categories = ['Linear Data', 'Spot Data', 'Both Available', 'Prices Available']
    rates = [
        df['linear_data_available'].mean(),
        df['spot_data_available'].mean(), 
        df['both_available'].mean(),
        df['prices_available'].mean()
    ]
    
    if df['bitcoin_proxy_used'].any():
        categories.append('Final Success\n(w/ Proxy)')
        rates.append(df['final_success'].mean() if 'final_success' in df.columns else df['both_available'].mean())
    
    bars = ax1.bar(categories, rates, color=['skyblue', 'lightcoral', 'lightgreen', 'orange', 'gold'][:len(categories)])
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Data Availability Success Rates')
    ax1.set_ylim(0, 1)
    
    # Add percentage labels on bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.1%}', ha='center', va='bottom')
    
    # Plot 2: Symbol availability distribution
    ax2 = axes[0, 1]
    ax2.hist(symbol_availability['both_available'], bins=20, color='lightblue', alpha=0.7)
    ax2.set_xlabel('Data Availability Rate')
    ax2.set_ylabel('Number of Symbols')
    ax2.set_title('Distribution of Symbol Data Availability')
    
    # Plot 3: Availability over time
    ax3 = axes[1, 0]
    if 'rebalance_date' in df.columns and len(time_availability) > 1:
        ax3.plot(time_availability['rebalance_date'], time_availability['both_available'], 'b-o', label='Both Available')
        ax3.plot(time_availability['rebalance_date'], time_availability['linear_data_available'], 'g-s', label='Linear Available')
        ax3.plot(time_availability['rebalance_date'], time_availability['spot_data_available'], 'r-^', label='Spot Available')
        ax3.set_ylabel('Success Rate')
        ax3.set_title('Data Availability Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax3.text(0.5, 0.5, 'Insufficient time series data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Data Availability Over Time')
    
    # Plot 4: Bitcoin proxy usage
    ax4 = axes[1, 1]
    if df['bitcoin_proxy_attempted'].any():
        proxy_data = df[df['bitcoin_proxy_attempted']]
        proxy_success = proxy_data['bitcoin_proxy_used'].sum()
        proxy_total = len(proxy_data)
        proxy_fail = proxy_total - proxy_success
        
        ax4.pie([proxy_success, proxy_fail], labels=['Successful', 'Failed'], 
               colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%')
        ax4.set_title(f'Bitcoin Proxy Success Rate\n({proxy_total} attempts)')
    else:
        ax4.text(0.5, 0.5, 'No Bitcoin proxy\nattempts made', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Bitcoin Proxy Usage')
    
    plt.tight_layout()
    
    # Save the plot
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    plot_path = f"{output_dir}/data_availability_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n📈 Data availability analysis saved to {plot_path}")
    
    plt.show()
    
    # Save detailed CSV report
    symbol_availability_path = f"{output_dir}/symbol_data_availability.csv"
    symbol_availability.to_csv(symbol_availability_path, index=False)
    
    if 'rebalance_date' in df.columns:
        time_availability_path = f"{output_dir}/time_data_availability.csv"
        time_availability.to_csv(time_availability_path, index=False)
        print(f"📊 Time-based availability data saved to {time_availability_path}")
    
    print(f"📊 Symbol-based availability data saved to {symbol_availability_path}")
    
    return symbol_availability, time_availability if 'rebalance_date' in df.columns else None


def get_data_availability_summary(data_availability):
    """
    Get a quick summary of data availability
    
    Args:
        data_availability (list): List of data availability records
        
    Returns:
        dict: Summary statistics
    """
    import pandas as pd
    
    if not data_availability:
        return {}
    
    df = pd.DataFrame(data_availability)
    
    # Add missing columns if they don't exist
    required_columns = ['bitcoin_proxy_used', 'bitcoin_proxy_attempted']
    for col in required_columns:
        if col not in df.columns:
            df[col] = False
    
    summary = {
        'total_evaluations': len(df),
        'unique_symbols': df['symbol'].nunique(),
        'linear_success_rate': df['linear_data_available'].mean(),
        'spot_success_rate': df['spot_data_available'].mean(),
        'both_success_rate': df['both_available'].mean(),
        'price_success_rate': df['prices_available'].mean(),
        'proxy_attempts': df['bitcoin_proxy_attempted'].sum(),
        'proxy_successes': df['bitcoin_proxy_used'].sum(),
        'proxy_success_rate': df['bitcoin_proxy_used'].sum() / max(df['bitcoin_proxy_attempted'].sum(), 1)
    }
    
    # Calculate final success rate including proxy
    if df['bitcoin_proxy_used'].any():
        df['final_success'] = df['both_available'] | df['bitcoin_proxy_used'] 
        summary['final_success_rate'] = df['final_success'].mean()
    else:
        summary['final_success_rate'] = summary['both_success_rate']
    
    return summary

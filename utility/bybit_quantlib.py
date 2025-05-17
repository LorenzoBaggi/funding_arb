import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pybit.unified_trading import HTTP
import time
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
from tqdm import tqdm 
import pytz  
from scipy import stats

def load_symbols(csv_path="symbols.csv"):
    """
    Load symbols from the CSV file
    
    Args:
        csv_path (str): Path to the CSV file containing symbols
        
    Returns:
        pd.DataFrame: DataFrame containing the symbols
    """
    df = pd.read_csv(csv_path)
    # Convert launchTime column to datetime if it's a string
    if 'launchTime' in df.columns and isinstance(df['launchTime'].iloc[0], str):
        df['launchTime'] = pd.to_datetime(df['launchTime'])
        # Add timezone info if not present
        if df['launchTime'].dt.tz is None:
            df['launchTime'] = df['launchTime'].dt.tz_localize(pytz.UTC)
    return df

def fetch_funding_rate_history(symbol, start_time=None, end_time=None, save_path=None):
    """
    Fetches historical funding rate data for a specific symbol from Bybit
    
    Args:
        symbol (str): The trading pair symbol (e.g., 'BTCUSDT')
        start_time (int, optional): Start timestamp in milliseconds
        end_time (int, optional): End timestamp in milliseconds
        save_path (str, optional): Path to save the CSV file
        
    Returns:
        pd.DataFrame: DataFrame containing the funding rate history
    """
    # Initialize HTTP session
    session = HTTP()
    
    # Initialize parameters dictionary
    params = {
        "category": "linear",  # or "inverse" for inverse contracts
        "symbol": symbol,
        "limit": 200  # maximum number of records per request
    }
    
    # Add time parameters if provided
    if end_time:
        params["endTime"] = end_time
    if start_time and end_time:
        params["startTime"] = start_time
    
    all_funding_data = []
    
    # Check if we need to make multiple requests to get all data
    if start_time and end_time:
        current_end_time = end_time
        
        while current_end_time > start_time:
            try:
                # Fetch data
                utc_time = datetime.fromtimestamp(current_end_time/1000).replace(tzinfo=pytz.UTC)
                print(f"Fetching data before {utc_time}")
                params["endTime"] = current_end_time
                result = session.get_funding_rate_history(**params)
                
                # Check if we got data
                if result["retCode"] == 0 and result["result"]["list"]:
                    funding_data = result["result"]["list"]
                    all_funding_data.extend(funding_data)
                    
                    # Update the end time for the next request
                    oldest_timestamp = min(int(item["fundingRateTimestamp"]) for item in funding_data)
                    current_end_time = oldest_timestamp - 1
                else:
                    print("No more data or error in response")
                    break
                
                # Be nice to the API rate limits
                time.sleep(0.2)
                
            except Exception as e:
                print(f"Error fetching data: {e}")
                break
    else:
        # Single request
        try:
            result = session.get_funding_rate_history(**params)
            if result["retCode"] == 0:
                all_funding_data = result["result"]["list"]
        except Exception as e:
            print(f"Error fetching data: {e}")
    
    # Convert to DataFrame
    if all_funding_data:
        df = pd.DataFrame(all_funding_data)
        
        # Convert timestamp to datetime with UTC timezone
        df["fundingRateTimestamp"] = pd.to_datetime(df["fundingRateTimestamp"], unit='ms')
        if df["fundingRateTimestamp"].dt.tz is None:
            df["fundingRateTimestamp"] = df["fundingRateTimestamp"].dt.tz_localize(pytz.UTC)
        
        # Convert funding rate to float
        df["fundingRate"] = df["fundingRate"].astype(float)
        
        # Sort by timestamp
        df = df.sort_values("fundingRateTimestamp")
        
        # Save to CSV if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # Convert timezone-aware datetime to string before saving to CSV
            df_save = df.copy()
            df_save["fundingRateTimestamp"] = df_save["fundingRateTimestamp"].dt.strftime('%Y-%m-%d %H:%M:%S%z')
            df_save.to_csv(save_path, index=False)
            print(f"Data saved to {save_path}")
        
        return df
    
    return pd.DataFrame()

def fetch_all_funding_rates(symbols_df, start_date, end_date, output_dir="funding_data"):
    """
    Fetch funding rates for all symbols within the specified period
    
    Args:
        symbols_df (pd.DataFrame): DataFrame with symbols including fundingInterval
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        output_dir (str): Directory to save CSV files
        
    Returns:
        dict: Dictionary with symbols as keys and funding rate DataFrames as values
    """
    # Convert dates to timestamps (UTC)
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=pytz.UTC)
    start_timestamp = int(start_dt.timestamp() * 1000)
    
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    if end_dt.tzinfo is None:
        end_dt = end_dt.replace(tzinfo=pytz.UTC)
    end_timestamp = int(end_dt.timestamp() * 1000)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    funding_data = {}
    
    # Iterate over all symbols with a progress bar
    for _, row in tqdm(symbols_df.iterrows(), total=len(symbols_df), desc="Fetching funding rates"):
        symbol = row['symbol']
        
        # Get the funding interval for this symbol (in minutes)
        funding_interval = row.get('fundingInterval', 480)  # Default to 8h (480 min) if not available
        
        # Check if launch time is after the requested start date
        if pd.notna(row['launchTime']):
            # Ensure launchTime is in UTC
            launch_time = row['launchTime']
            if launch_time.tzinfo is None:
                launch_time = launch_time.tz_localize(pytz.UTC)
                
            symbol_launch_timestamp = int(launch_time.timestamp() * 1000)
            # Adjust start date if necessary
            symbol_start_timestamp = max(start_timestamp, symbol_launch_timestamp)
        else:
            symbol_start_timestamp = start_timestamp
        
        # Skip this symbol if start date is after end date
        if symbol_start_timestamp >= end_timestamp:
            print(f"Skipping {symbol} - launched after requested end date")
            continue
        
        # CSV file path for this symbol
        csv_path = f"{output_dir}/{symbol}_funding.csv"
        
        # Check if file already exists
        if os.path.exists(csv_path):
            # Load existing data
            df = pd.read_csv(csv_path)
            df['fundingRateTimestamp'] = pd.to_datetime(df['fundingRateTimestamp'])
            # Add timezone info if not present
            if df['fundingRateTimestamp'].dt.tz is None:
                df['fundingRateTimestamp'] = df['fundingRateTimestamp'].dt.tz_localize(pytz.UTC)
            print(f"Loaded existing data for {symbol}")
        else:
            # Fetch data from Bybit
            print(f"Fetching data for {symbol}...")
            df = fetch_funding_rate_history(symbol, symbol_start_timestamp, end_timestamp, csv_path)
        
        if not df.empty:
            # Add metadata about this symbol to the DataFrame
            df = df.copy()  # Make a copy to avoid SettingWithCopyWarning
            df['symbol'] = symbol
            df['fundingInterval'] = funding_interval
            funding_data[symbol] = df
    
    return funding_data

def calculate_daily_mean_funding(funding_data):
    """
    Calculate the daily mean funding rate across all symbols,
    properly handling different funding intervals
    
    Args:
        funding_data (dict): Dictionary with funding rate DataFrames
        
    Returns:
        pd.DataFrame: DataFrame with daily mean funding rates
    """
    # List to store daily DataFrames
    daily_dfs = []
    
    for symbol, df in funding_data.items():
        try:
            # Ensure we have data for this symbol
            if df.empty:
                continue
                
            # Convert timestamp column to datetime with UTC timezone
            df['fundingRateTimestamp'] = pd.to_datetime(df['fundingRateTimestamp'])
            
            # Safely handle timezone
            if df['fundingRateTimestamp'].dt.tz is None:
                df['fundingRateTimestamp'] = df['fundingRateTimestamp'].dt.tz_localize(pytz.UTC)
            elif df['fundingRateTimestamp'].dt.tz != pytz.UTC:
                df['fundingRateTimestamp'] = df['fundingRateTimestamp'].dt.tz_convert(pytz.UTC)
            
            # Extract date (without time) in UTC
            df['date'] = df['fundingRateTimestamp'].dt.date
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(pytz.UTC)
            
            # Get funding interval for this symbol (in minutes)
            funding_interval = df['fundingInterval'].iloc[0]
            
            # Calculate normalization factor (how many funding events per day)
            # 1440 minutes in a day
            normalization_factor = 1440 / funding_interval
            
            # Normalize funding rate to daily equivalent
            df['normalized_rate'] = df['fundingRate'] * normalization_factor
            
            # Calculate daily average for each symbol 
            daily_mean = df.groupby('date').agg({
                'fundingRate': 'mean',        # Original per-interval rate
                'normalized_rate': 'mean',    # Normalized daily rate
                'fundingInterval': 'first'    # Keep the interval information
            }).reset_index()
            
            daily_mean['symbol'] = symbol
            
            daily_dfs.append(daily_mean)
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            continue
    
    # Merge all DataFrames
    if not daily_dfs:
        return pd.DataFrame()
    
    all_daily = pd.concat(daily_dfs, ignore_index=True)
    
    # Calculate mean across all symbols for each day 
    # (both normalized and unnormalized for comparison)
    daily_mean = all_daily.groupby('date').agg({
        'fundingRate': 'mean',        # Simple average of per-interval rates
        'normalized_rate': 'mean'     # Average of normalized daily rates
    }).reset_index()
    
    # Rename for clarity
    daily_mean.rename(columns={
        'fundingRate': 'mean_rate', 
        'normalized_rate': 'normalized_mean_rate'
    }, inplace=True)
    
    # Calculate number of available symbols per day
    symbol_count = all_daily.groupby('date')['symbol'].nunique().reset_index()
    symbol_count.rename(columns={'symbol': 'symbol_count'}, inplace=True)
    
    # Merge with the main DataFrame
    daily_mean = pd.merge(daily_mean, symbol_count, on='date')
    
    # Sort by date
    daily_mean = daily_mean.sort_values('date')
    
    return daily_mean

def calculate_moving_averages_and_future_funding(daily_mean, window=7, remove_overlapping=True, rate_column='normalized_mean_rate'):
    """
    Calculate moving averages of funding rates and future average funding rates
    
    Args:
        daily_mean (pd.DataFrame): DataFrame with daily mean funding rates
        window (int): Window size for moving average
        remove_overlapping (bool): Whether to remove overlapping periods
        rate_column (str): Column to use for calculations ('mean_rate' or 'normalized_mean_rate')
        
    Returns:
        pd.DataFrame: DataFrame with moving averages and future funding rates
    """
    # Make a copy to avoid modifying the original
    result_df = daily_mean.copy()
    
    # Ensure date is timezone-aware
    if result_df['date'].dt.tz is None:
        result_df['date'] = result_df['date'].dt.tz_localize(pytz.UTC)
    
    # Calculate rolling window statistics
    result_df['rolling_avg'] = result_df[rate_column].rolling(window=window).mean()
    result_df['rolling_std'] = result_df[rate_column].rolling(window=window).std()
    result_df['rolling_min'] = result_df[rate_column].rolling(window=window).min()
    result_df['rolling_max'] = result_df[rate_column].rolling(window=window).max()
    
    # Calculate future funding
    future_values = []
    
    for i in range(len(result_df)):
        if i + window < len(result_df):
            # Future average
            future_avg = result_df[rate_column].iloc[i+1:i+window+1].mean()
            # Future standard deviation
            future_std = result_df[rate_column].iloc[i+1:i+window+1].std()
            # Future min/max
            future_min = result_df[rate_column].iloc[i+1:i+window+1].min()
            future_max = result_df[rate_column].iloc[i+1:i+window+1].max()
        else:
            future_avg, future_std, future_min, future_max = np.nan, np.nan, np.nan, np.nan
            
        future_values.append({
            'future_avg': future_avg,
            'future_std': future_std,
            'future_min': future_min, 
            'future_max': future_max
        })
    
    future_df = pd.DataFrame(future_values)
    result_df = pd.concat([result_df, future_df], axis=1)
    
    # Remove rows where rolling_avg or future_avg are NaN
    result_df = result_df.dropna(subset=['rolling_avg', 'future_avg'])
    
    # Remove overlapping periods if requested
    if remove_overlapping:
        # Keep only every 'window' data points to avoid overlapping
        result_df = result_df.iloc[::window].copy()
        print(f"Removed overlapping periods: reduced from {len(daily_mean)} to {len(result_df)} data points")
    
    return result_df

def plot_funding_rate_persistence(analysis_df, output_path="funding_persistence.png", remove_outliers=True, outlier_threshold=2.0, window=7):
    """
    Create a comprehensive scatter plot to visualize funding rate persistence
    with enhanced legends, annotations, and optional outlier removal
    
    Args:
        analysis_df (pd.DataFrame): DataFrame with moving averages and future funding rates
        output_path (str): Path to save the plot
        remove_outliers (bool): Whether to remove outliers for cleaner visualization
        outlier_threshold (float): Z-score threshold to identify outliers (default: 2.0)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy import stats
    
    # Make a copy to avoid modifying the original
    plot_df = analysis_df.copy()
    
    # Optionally remove outliers using Z-score method
    if remove_outliers:
        # Calculate z-scores for both axes
        z_score_x = np.abs(stats.zscore(plot_df['rolling_avg'], nan_policy='omit'))
        z_score_y = np.abs(stats.zscore(plot_df['future_avg'], nan_policy='omit'))
        
        # Identify non-outlier rows (where z-score is below threshold for both axes)
        non_outliers = (z_score_x < outlier_threshold) & (z_score_y < outlier_threshold)
        
        # Count outliers removed
        outliers_removed = len(plot_df) - np.sum(non_outliers)
        
        # Filter out outliers
        plot_df = plot_df[non_outliers].reset_index(drop=True)
        
        print(f"Removed {outliers_removed} outliers using Z-score threshold of {outlier_threshold}")
    
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot with symbols count color coding
    scatter = plt.scatter(
        plot_df['rolling_avg'], 
        plot_df['future_avg'], 
        alpha=0.8,
        c=plot_df['symbol_count'],
        cmap='viridis',
        s=80,                     # Larger point size for better visibility
        edgecolor='black',        # Black edge for better contrast
        linewidth=0.5             # Thin edge line
    )
    
    # Add dates as annotations for key points using a more balanced approach
    # We'll select points that represent different areas of the chart
    
    # Strategy 1: Select points by quadrant (positive/negative x/y values)
    quadrants = {
        'top_right': (plot_df['rolling_avg'] > 0) & (plot_df['future_avg'] > 0),
        'top_left': (plot_df['rolling_avg'] < 0) & (plot_df['future_avg'] > 0),
        'bottom_left': (plot_df['rolling_avg'] < 0) & (plot_df['future_avg'] < 0),
        'bottom_right': (plot_df['rolling_avg'] > 0) & (plot_df['future_avg'] < 0)
    }
    
    selected_indices = []
    
    # For each quadrant, try to select at least one point with the maximum absolute value
    for quadrant_filter in quadrants.values():
        if quadrant_filter.sum() > 0:
            quadrant_df = plot_df[quadrant_filter]
            # Select point with max sum of absolute values (combined extremity)
            extremity = quadrant_df['rolling_avg'].abs() + quadrant_df['future_avg'].abs()
            max_idx = extremity.idxmax()
            selected_indices.append(max_idx)
    
    # Strategy 2: Also select points with extreme values on either axis
    # Get points with extremes on x-axis
    top_x = plot_df['rolling_avg'].abs().nlargest(2).index.tolist()
    # Get points with extremes on y-axis
    top_y = plot_df['future_avg'].abs().nlargest(2).index.tolist()
    
    # Add these points to our selection, avoiding duplicates
    selected_indices.extend([idx for idx in top_x + top_y if idx not in selected_indices])
    
    # Limit to a reasonable number (e.g., 7 points max)
    selected_indices = selected_indices[:7]
    
    # Now use these indices to annotate the plot
    for idx in selected_indices:
        row = plot_df.iloc[idx]
        plt.annotate(
            row['date'].strftime('%Y-%m-%d'),  # Format date as string
            (row['rolling_avg'], row['future_avg']),
            xytext=(10, 5),                    # Offset text
            textcoords='offset points',
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
        )
    
    # Add regression line
    z = np.polyfit(plot_df['rolling_avg'], plot_df['future_avg'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(plot_df['rolling_avg'].min(), plot_df['rolling_avg'].max(), 100)
    plt.plot(x_range, p(x_range), 'b-', linewidth=2.5, label=f'Regression Line (slope={z[0]:.3f})')
    
    # Add 45-degree line for reference
    min_val = min(plot_df['rolling_avg'].min(), plot_df['future_avg'].min())
    max_val = max(plot_df['rolling_avg'].max(), plot_df['future_avg'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, 
             linewidth=1.5, label='Perfect Persistence (45° line)')
    
    # Calculate correlation coefficient and other statistics
    corr = plot_df['rolling_avg'].corr(plot_df['future_avg'])
    
    # Calculate percentage of points in same direction (both positive or both negative)
    same_direction = np.mean((plot_df['rolling_avg'] > 0) == (plot_df['future_avg'] > 0))
    same_direction_pct = same_direction * 100
    
    # Add detailed stats as text box
    stats_text = (
        f"Statistics:\n"
        f"Correlation: {corr:.3f}\n"
        f"Same Direction: {same_direction_pct:.1f}%\n"
        f"Data Points: {len(plot_df)}"
    )
    
    if remove_outliers:
        stats_text += f"\n(Removed {outliers_removed} outliers)"
        
    stats_text += (    
        f"\nDate Range: {plot_df['date'].min().strftime('%Y-%m-%d')} to {plot_df['date'].max().strftime('%Y-%m-%d')}\n"
        f"Window Size: {window} days"
    )
    
    plt.annotate(
        stats_text, 
        xy=(0.02, 0.02),               # Position in plot coordinates (lower left)
        xycoords='axes fraction',      # Use axis relative coordinates
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
        fontsize=10
    )
    
    # Add explanatory text about what the chart shows
    explanation_text = (
        "This chart shows the relationship between current funding rates\n"
        f"({window}-day moving average) and future funding rates (next {window} days).\n"
        "Points along the 45° line indicate perfect persistence of funding rates.\n"
        "Color indicates number of cryptocurrency symbols analyzed at each point."
    )
    
    plt.annotate(
        explanation_text, 
        xy=(0.02, 0.95),               # Position in plot coordinates (upper left)
        xycoords='axes fraction',      # Use axis relative coordinates
        va='top',                      # Vertical alignment
        bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", alpha=0.8),
        fontsize=10
    )
    
    # Add a colorbar to show symbol count
    cbar = plt.colorbar(scatter)
    cbar.set_label('Number of Cryptocurrency Symbols', fontsize=12)
    
    # Add grid and improve its appearance
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add plot details with improved styling
    plt.title(f'Funding Rate Persistence (Correlation = {corr:.3f})', fontsize=18, pad=20)
    plt.xlabel(f'Current Funding Rate ({window}-day Moving Average)', fontsize=14, labelpad=10)
    plt.ylabel(f'Future Funding Rate (Next {window} days)', fontsize=14, labelpad=10)
    
    # Format tick labels to show percentages (funding rates are usually shown as %)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.3f}%'))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.3f}%'))
    
    # Add legend
    plt.legend(loc='upper left', fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)  # Higher DPI for better quality
    plt.show()
    
    print(f"Enhanced plot saved as {output_path}")
    
    # Create a second informative plot: time series of funding rates
    plt.figure(figsize=(12, 6))
    
    # Use date index for this plot
    time_df = plot_df.set_index('date').sort_index()
    
    # Plot time series of both current and future rates
    plt.plot(time_df.index, time_df['rolling_avg']*100, 'b-', 
             label=f'Current Funding Rate ({window}-day MA)', linewidth=1.5)
    plt.plot(time_df.index, time_df['future_avg']*100, 'r--', 
             label=f'Future Funding Rate (Next {window} days)', linewidth=1.5)
    
    # Add zero line for reference
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add labels and title
    plt.title('Funding Rate Time Series', fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Funding Rate (%)', fontsize=14)
    
    # Add legend
    plt.legend(loc='best', fontsize=12)
    
    # Save time series plot
    time_series_path = output_path.replace('.png', '_time_series.png')
    plt.tight_layout()
    plt.savefig(time_series_path, dpi=300)
    plt.show()
    
    print(f"Time series plot saved as {time_series_path}")
    
    return plot_df  # Return the filtered dataframe for further analysis

def analyze_funding_spread_by_symbol(funding_data, symbols_df=None, output_dir="funding_analysis", days_to_analyze=7):
    """
    Analyze the absolute spread of funding rates for each symbol over a limited recent period,
    normalizing for different funding intervals
    
    Args:
        funding_data (dict): Dictionary with symbols as keys and funding rate DataFrames as values
        symbols_df (pd.DataFrame, optional): DataFrame with symbols metadata
        output_dir (str): Directory to save results
        days_to_analyze (int): Number of days to analyze (default: 7)
    """
  
    
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store statistics for each symbol
    symbol_stats = {}
    
    print(f"Analyzing funding rate spread by symbol for the last {days_to_analyze} days...")
    
    # Calculate the cutoff date
    current_time = datetime.now(pytz.UTC)
    cutoff_date = current_time - timedelta(days=days_to_analyze)
    
    # Process each symbol
    for symbol, df in funding_data.items():
        # Skip if DataFrame is empty
        if df.empty:
            continue
            
        # Ensure timestamps are timezone-aware
        if df['fundingRateTimestamp'].dt.tz is None:
            df['fundingRateTimestamp'] = df['fundingRateTimestamp'].dt.tz_localize(pytz.UTC)
        
        # Filter data to only include the last N days
        df_recent = df[df['fundingRateTimestamp'] >= cutoff_date]
        
        # Skip if recent DataFrame is empty
        if df_recent.empty:
            continue
            
        # Get funding interval for this symbol (in minutes)
        funding_interval = df_recent['fundingInterval'].iloc[0]
        
        # Calculate normalization factor (how many funding events per day)
        # 1440 minutes in a day
        normalization_factor = 1440 / funding_interval
        
        # Ensure we're working with numeric data
        df_recent['fundingRate'] = pd.to_numeric(df_recent['fundingRate'], errors='coerce')
        
        # Calculate absolute funding rate for each entry
        df_recent['abs_funding_rate'] = df_recent['fundingRate'].abs()
        
        # Calculate normalized rates (daily equivalent)
        df_recent['normalized_rate'] = df_recent['fundingRate'] * normalization_factor
        df_recent['normalized_abs_rate'] = df_recent['abs_funding_rate'] * normalization_factor
        
        # Calculate statistics for the last N days
        stats = {
            'symbol': symbol,
            'funding_interval_minutes': funding_interval,
            'funding_events_per_day': normalization_factor,
            
            # Original rates (per interval)
            'mean_rate_per_interval': df_recent['fundingRate'].mean() * 100,  # Convert to percentage
            'mean_abs_rate_per_interval': df_recent['abs_funding_rate'].mean() * 100,
            'std_rate_per_interval': df_recent['fundingRate'].std() * 100,
            
            # Normalized rates (daily equivalent)
            'mean_daily_rate': df_recent['normalized_rate'].mean() * 100,
            'mean_daily_abs_rate': df_recent['normalized_abs_rate'].mean() * 100,  # This is our key metric
            'std_daily_rate': df_recent['normalized_rate'].std() * 100,
            
            # Other statistics
            'max_abs_rate': df_recent['abs_funding_rate'].max() * 100,
            'max_daily_abs_rate': df_recent['normalized_abs_rate'].max() * 100,
            'count': len(df_recent),
            'positive_count': (df_recent['fundingRate'] > 0).sum(),
            'negative_count': (df_recent['fundingRate'] < 0).sum(),
            'positive_pct': (df_recent['fundingRate'] > 0).mean() * 100,
            'mean_positive': df_recent.loc[df_recent['fundingRate'] > 0, 'fundingRate'].mean() * 100 if (df_recent['fundingRate'] > 0).any() else 0,
            'mean_negative': df_recent.loc[df_recent['fundingRate'] < 0, 'fundingRate'].mean() * 100 if (df_recent['fundingRate'] < 0).any() else 0,
            'first_date': df_recent['fundingRateTimestamp'].min(),
            'last_date': df_recent['fundingRateTimestamp'].max(),
        }
        
        # Add max timestamp and value
        if not df_recent['abs_funding_rate'].isna().all():
            max_idx = df_recent['abs_funding_rate'].idxmax()
            stats['max_timestamp'] = df_recent.loc[max_idx, 'fundingRateTimestamp']
            stats['max_rate_value'] = df_recent.loc[max_idx, 'fundingRate'] * 100
        else:
            stats['max_timestamp'] = None
            stats['max_rate_value'] = np.nan
        
        # Calculate sharpe-like ratio (mean/std) only if std > 0
        if stats['std_daily_rate'] > 0:
            stats['sharpe_ratio'] = stats['mean_daily_abs_rate'] / stats['std_daily_rate']
        else:
            stats['sharpe_ratio'] = 0
            
        # Calculate annualized returns (assuming compound growth)
        # Daily rate to annual rate: (1 + daily_rate)^365 - 1
        if stats['mean_daily_rate'] != 0:
            daily_rate_decimal = stats['mean_daily_rate'] / 100  # Convert from percentage to decimal
            stats['annual_return'] = ((1 + daily_rate_decimal) ** 365 - 1) * 100  # Convert back to percentage
        else:
            stats['annual_return'] = 0
            
        symbol_stats[symbol] = stats
    
    # Convert to DataFrame for easier manipulation
    stats_df = pd.DataFrame(list(symbol_stats.values()))
    
    # Check if there are enough symbols with data
    if len(stats_df) == 0:
        print(f"No symbols found with data in the last {days_to_analyze} days")
        return pd.DataFrame()
    
    # Filter out symbols with very few data points (may be unreliable)
    min_count = min(10, days_to_analyze)  # Adjust minimum count based on days analyzed
    filtered_stats = stats_df[stats_df['count'] >= min_count].copy()
    
    print(f"Analyzed {len(filtered_stats)} symbols with at least {min_count} data points in the last {days_to_analyze} days")
    
    # Check if there are still symbols after filtering
    if len(filtered_stats) == 0:
        print(f"No symbols found with at least {min_count} data points in the last {days_to_analyze} days")
        return pd.DataFrame()
    
    # Sort by NORMALIZED mean absolute spread (descending)
    spread_sorted = filtered_stats.sort_values('mean_daily_abs_rate', ascending=False).reset_index(drop=True)
    
    # Sort by max absolute spread (descending)
    max_sorted = filtered_stats.sort_values('max_daily_abs_rate', ascending=False).reset_index(drop=True)
    
    # Sort by Sharpe ratio (descending)
    sharpe_sorted = filtered_stats.sort_values('sharpe_ratio', ascending=False).reset_index(drop=True)
    
    # Sort by annual return (descending)
    return_sorted = filtered_stats.sort_values('annual_return', ascending=False).reset_index(drop=True)
    
    # Save the full statistics to CSV
    stats_path = f"{output_dir}/funding_rate_stats_by_symbol_last_{days_to_analyze}days.csv"
    filtered_stats.to_csv(stats_path, index=False)
    print(f"Symbol statistics saved to {stats_path}")
    
    # Create the absolute spread bar chart (NORMALIZED TO DAILY RATES)
    plt.figure(figsize=(16, 8))
    
    # Get the top 50 symbols by mean absolute spread (or all if fewer than 50)
    top_n = min(50, len(spread_sorted))
    top_symbols_spread = spread_sorted.head(top_n)
    
    # Create bar chart for absolute spread
    bars = plt.bar(
        range(len(top_symbols_spread)), 
        top_symbols_spread['mean_daily_abs_rate'],
        color='skyblue',
        alpha=0.8
    )
    
    # Color bars by funding interval
    interval_colors = {
        60: 'lightcoral',    # 1h
        120: 'lightsalmon',  # 2h
        240: 'lightblue',    # 4h
        480: 'skyblue',      # 8h
        720: 'lightgreen',   # 12h
        1440: 'lightgray'    # 24h
    }
    
    # Set bar colors based on funding interval
    for i, bar in enumerate(bars):
        interval = top_symbols_spread.iloc[i]['funding_interval_minutes']
        bar.set_color(interval_colors.get(interval, 'skyblue'))
    
    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Set x-axis labels to symbol names
    plt.xticks(
        range(len(top_symbols_spread)), 
        top_symbols_spread['symbol'].str.replace('USDT', ''),  # Remove USDT suffix for cleaner labels
        rotation=90
    )
    
    # Add labels and title
    plt.xlabel('Coins', fontsize=12)
    plt.ylabel('Average Daily Equivalent Absolute Spread (%)', fontsize=12)
    plt.title(f'Average Daily Equivalent Absolute Spread (Last {days_to_analyze} Days)', fontsize=16)
    
    # Create a legend for funding intervals
    legend_patches = [mpatches.Patch(color=color, label=f"{interval//60}h interval") 
                     for interval, color in interval_colors.items()]
    plt.legend(handles=legend_patches, loc='upper right')
    
    # Add value labels on top of each bar
    for i, value in enumerate(top_symbols_spread['mean_daily_abs_rate']):
        if value > 5:  # Only add text for bars with significant values
            plt.text(
                i, value + 2,  # Position slightly above the bar
                f'{value:.1f}%',
                ha='center',
                fontsize=8,
                rotation=0
            )
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{output_dir}/average_daily_absolute_spread_last_{days_to_analyze}days.png", dpi=300)
    plt.show()
    
    # Create a bar chart for maximum funding rates
    plt.figure(figsize=(16, 8))
    
    # Get the top 50 symbols by max absolute spread
    top_n_max = min(50, len(max_sorted))
    top_max_spread = max_sorted.head(top_n_max)
    
    # Create bar chart for max absolute spread
    bars_max = plt.bar(
        range(len(top_max_spread)), 
        top_max_spread['max_daily_abs_rate'],
        color='salmon',
        alpha=0.8
    )
    
    # Set bar colors based on funding interval
    for i, bar in enumerate(bars_max):
        interval = top_max_spread.iloc[i]['funding_interval_minutes']
        bar.set_color(interval_colors.get(interval, 'salmon'))
    
    # Add grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Set x-axis labels to symbol names
    plt.xticks(
        range(len(top_max_spread)), 
        top_max_spread['symbol'].str.replace('USDT', ''),  # Remove USDT suffix for cleaner labels
        rotation=90
    )
    
    # Add labels and title
    plt.xlabel('Coins', fontsize=12)
    plt.ylabel('Maximum Daily Equivalent Absolute Spread (%)', fontsize=12)
    plt.title(f'Maximum Daily Equivalent Absolute Spread (Last {days_to_analyze} Days)', fontsize=16)
    
    # Create a legend for funding intervals
    plt.legend(handles=legend_patches, loc='upper right')
    
    # Add value labels on top of each bar
    for i, value in enumerate(top_max_spread['max_daily_abs_rate']):
        if value > 5:  # Only add text for bars with significant values
            plt.text(
                i, value + 2,  # Position slightly above the bar
                f'{value:.1f}%',
                ha='center',
                fontsize=8,
                rotation=0
            )
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{output_dir}/max_daily_absolute_spread_last_{days_to_analyze}days.png", dpi=300)
    plt.show()
    
    # Create a table showing normalization factors
    plt.figure(figsize=(10, 6))
    
    # Group by funding interval and count
    interval_counts = filtered_stats.groupby('funding_interval_minutes').size().reset_index()
    interval_counts.columns = ['Interval (minutes)', 'Count']
    interval_counts['Interval (hours)'] = interval_counts['Interval (minutes)'] / 60
    interval_counts['Fundings per Day'] = 1440 / interval_counts['Interval (minutes)']
    interval_counts['Normalization Factor'] = interval_counts['Fundings per Day']
    
    # Reorder columns
    interval_counts = interval_counts[['Interval (minutes)', 'Interval (hours)', 
                                      'Fundings per Day', 'Normalization Factor', 'Count']]
    
    # Create a table
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False)
    
    table = plt.table(
        cellText=interval_counts.values.round(2),
        colLabels=interval_counts.columns,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    plt.title(f'Funding Interval Normalization Factors (Last {days_to_analyze} Days)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/funding_interval_normalization_last_{days_to_analyze}days.png", dpi=300)
    plt.show()
    
    # Create a comprehensive report with insights
    with open(f"{output_dir}/funding_spread_analysis_report_last_{days_to_analyze}days.txt", 'w') as f:
        f.write(f"FUNDING RATE SPREAD ANALYSIS REPORT (LAST {days_to_analyze} DAYS)\n")
        f.write("=============================================\n\n")
        
        f.write(f"Analysis Period: {filtered_stats['first_date'].min()} to {filtered_stats['last_date'].max()}\n")
        f.write(f"Total Symbols Analyzed: {len(filtered_stats)}\n\n")
        
        f.write("NORMALIZATION METHODOLOGY\n")
        f.write("------------------------\n")
        f.write("All funding rates have been normalized to their daily equivalent values for fair comparison.\n")
        f.write("For example:\n")
        f.write("- 8h interval (3 fundings per day): rate × 3 = daily equivalent\n")
        f.write("- 1h interval (24 fundings per day): rate × 24 = daily equivalent\n\n")
        
        f.write("FUNDING INTERVAL DISTRIBUTION\n")
        f.write("----------------------------\n")
        for interval, count in filtered_stats['funding_interval_minutes'].value_counts().sort_index().items():
            f.write(f"{int(interval/60)}h interval: {count} symbols ({count/len(filtered_stats)*100:.1f}%)\n")
        
        f.write("\nTOP 10 SYMBOLS BY DAILY EQUIVALENT ABSOLUTE SPREAD\n")
        f.write("----------------------------------------------\n")
        for i, row in spread_sorted.head(10).iterrows():
            f.write(f"{i+1}. {row['symbol']} ({int(row['funding_interval_minutes']/60)}h): " +
                    f"{row['mean_daily_abs_rate']:.2f}% daily equiv. " +
                    f"(Interval rate: {row['mean_abs_rate_per_interval']:.4f}%, " +
                    f"Sharpe: {row['sharpe_ratio']:.2f}, " +
                    f"Positive: {row['positive_pct']:.1f}%)\n")
        
        f.write("\nTOP 10 SYMBOLS BY MAXIMUM DAILY EQUIVALENT ABSOLUTE SPREAD\n")
        f.write("----------------------------------------------\n")
        for i, row in max_sorted.head(10).iterrows():
            max_date = row['max_timestamp']
            max_date_str = max_date.strftime('%Y-%m-%d %H:%M:%S') if max_date is not None else "N/A"
            max_rate_value = row['max_rate_value']
            max_rate_sign = "+" if max_rate_value >= 0 else ""
            
            f.write(f"{i+1}. {row['symbol']} ({int(row['funding_interval_minutes']/60)}h): " +
                    f"{row['max_daily_abs_rate']:.2f}% max daily equiv. " +
                    f"(Actual rate: {max_rate_sign}{max_rate_value:.4f}%, " +
                    f"Date: {max_date_str}, " +
                    f"Avg: {row['mean_daily_abs_rate']:.2f}%)\n")
        
        f.write("\nTOP 10 SYMBOLS BY ANNUAL EQUIVALENT RETURN\n")
        f.write("---------------------------------------\n")
        for i, row in return_sorted.head(10).iterrows():
            f.write(f"{i+1}. {row['symbol']} ({int(row['funding_interval_minutes']/60)}h): " +
                    f"{row['annual_return']:.2f}% annual return " + 
                    f"(Daily: {row['mean_daily_rate']:.4f}%, " +
                    f"Direction: {'Positive' if row['mean_daily_rate'] > 0 else 'Negative'}, " +
                    f"Consistency: {max(row['positive_pct'], 100-row['positive_pct']):.1f}%)\n")
        
        f.write("\nTOP 10 SYMBOLS BY SHARPE RATIO\n")
        f.write("-----------------------------\n")
        for i, row in sharpe_sorted.head(10).iterrows():
            f.write(f"{i+1}. {row['symbol']} ({int(row['funding_interval_minutes']/60)}h): " +
                    f"Sharpe {row['sharpe_ratio']:.2f} " +
                    f"(Daily spread: {row['mean_daily_abs_rate']:.2f}%, " +
                    f"Positive: {row['positive_pct']:.1f}%)\n")
        
        f.write("\nTOP 10 MOST CONSISTENTLY POSITIVE\n")
        f.write("-------------------------------\n")
        positive_sorted = filtered_stats.sort_values('positive_pct', ascending=False).reset_index(drop=True)
        for i, row in positive_sorted.head(10).iterrows():
            f.write(f"{i+1}. {row['symbol']} ({int(row['funding_interval_minutes']/60)}h): " +
                    f"{row['positive_pct']:.1f}% positive " +
                    f"(Daily spread: {row['mean_daily_abs_rate']:.2f}%, " +
                    f"Annual equiv: {row['annual_return']:.2f}%)\n")
        
        f.write("\nTOP 10 MOST CONSISTENTLY NEGATIVE\n")
        f.write("-------------------------------\n")
        negative_sorted = filtered_stats.sort_values('positive_pct', ascending=True).reset_index(drop=True)
        for i, row in negative_sorted.head(10).iterrows():
            neg_pct = 100 - row['positive_pct']
            f.write(f"{i+1}. {row['symbol']} ({int(row['funding_interval_minutes']/60)}h): " +
                    f"{neg_pct:.1f}% negative " +
                    f"(Daily spread: {row['mean_daily_abs_rate']:.2f}%, " +
                    f"Annual equiv: {row['annual_return']:.2f}%)\n")
        
        f.write("\nKEY INSIGHTS\n")
        f.write("-----------\n")
        # Calculate some key insights
        avg_spread = filtered_stats['mean_daily_abs_rate'].mean()
        max_spread_symbol = spread_sorted.iloc[0]['symbol']
        max_spread_value = spread_sorted.iloc[0]['mean_daily_abs_rate']
        max_spread_interval = spread_sorted.iloc[0]['funding_interval_minutes']
        
        # Maximum spread insights
        max_value_symbol = max_sorted.iloc[0]['symbol']
        max_value = max_sorted.iloc[0]['max_daily_abs_rate']
        max_value_interval = max_sorted.iloc[0]['funding_interval_minutes']
        max_value_date = max_sorted.iloc[0]['max_timestamp']
        max_value_date_str = max_value_date.strftime('%Y-%m-%d %H:%M:%S') if max_value_date is not None else "N/A"
        
        f.write(f"- Average daily equivalent funding rate spread: {avg_spread:.2f}%\n")
        f.write(f"- Highest average spread symbol: {max_spread_symbol} " +
                f"({int(max_spread_interval/60)}h interval, {max_spread_value:.2f}% daily equiv.)\n")
        f.write(f"- Highest maximum spread symbol: {max_value_symbol} " +
                f"({int(max_value_interval/60)}h interval, {max_value:.2f}% max daily equiv. on {max_value_date_str})\n")
        
        # Analyze by interval
        f.write("\nAVERAGE SPREAD BY FUNDING INTERVAL\n")
        f.write("-------------------------------\n")
        interval_analysis = filtered_stats.groupby('funding_interval_minutes').agg({
            'mean_daily_abs_rate': 'mean',
            'sharpe_ratio': 'mean',
            'symbol': 'count'
        }).reset_index()
        
        for _, row in interval_analysis.sort_values('funding_interval_minutes').iterrows():
            f.write(f"{int(row['funding_interval_minutes']/60)}h interval: " +
                    f"{row['mean_daily_abs_rate']:.2f}% avg daily equiv. spread " +
                    f"(Avg Sharpe: {row['sharpe_ratio']:.2f}, Count: {row['symbol']})\n")
        
        # Potential arbitrage opportunities
        f.write("\nPOTENTIAL ARBITRAGE OPPORTUNITIES (NORMALIZED)\n")
        f.write("------------------------------------------\n")
        # High spread + consistent direction + good Sharpe
        arb_candidates = filtered_stats[
            (filtered_stats['mean_daily_abs_rate'] > avg_spread * 1.5) &  # 50% higher than average spread
            ((filtered_stats['positive_pct'] > 65) | (filtered_stats['positive_pct'] < 35)) &  # Consistent direction
            (filtered_stats['sharpe_ratio'] > 1.0)  # Decent Sharpe ratio
        ].sort_values('mean_daily_abs_rate', ascending=False)
        
        if len(arb_candidates) > 0:
            for i, row in arb_candidates.head(10).iterrows():
                direction = "positive" if row['positive_pct'] > 50 else "negative"
                consistency = row['positive_pct'] if direction == "positive" else (100 - row['positive_pct'])
                f.write(f"{i+1}. {row['symbol']} ({int(row['funding_interval_minutes']/60)}h): " +
                        f"{row['mean_daily_abs_rate']:.2f}% daily equiv. spread, " +
                        f"{consistency:.1f}% {direction}, " +
                        f"Annual equiv: {row['annual_return']:.2f}%, " +
                        f"Sharpe: {row['sharpe_ratio']:.2f}\n")
        else:
            f.write("No strong arbitrage candidates found based on current criteria.\n")
    
    print(f"Comprehensive normalized analysis for the last {days_to_analyze} days saved to {output_dir}/funding_spread_analysis_report_last_{days_to_analyze}days.txt")
    return filtered_stats

def plot_aggregated_funding_rates(funding_data, symbols_df=None, output_dir="funding_analysis", 
                                 start_date=None, end_date=None, 
                                 metric="mean", normalized=True):
    """
    Create a time series plot of aggregated funding rates (max or mean) across all symbols,
    normalized by their funding intervals.
    
    Args:
        funding_data (dict): Dictionary with symbols as keys and funding rate DataFrames as values
        symbols_df (pd.DataFrame, optional): DataFrame with symbols metadata including fundingInterval
        output_dir (str): Directory to save results
        start_date (str, optional): Start date in 'YYYY-MM-DD' format
        end_date (str, optional): End date in 'YYYY-MM-DD' format
        metric (str): 'mean' or 'max' - which aggregation to use for the daily rates
        normalized (bool): Whether to normalize rates by funding interval
        
    Returns:
        pd.DataFrame: DataFrame with daily aggregated funding rates
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Convert date strings to datetime objects if provided
    if start_date:
        start_date = pd.to_datetime(start_date).tz_localize(pytz.UTC)
    if end_date:
        end_date = pd.to_datetime(end_date).tz_localize(pytz.UTC)
    
    print("Processing funding rate data from all symbols...")
    
    # Prepare a list to hold all normalized funding data
    all_funding_data = []
    
    # Process each symbol
    for symbol, df in funding_data.items():
        if df.empty:
            continue
        
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Ensure timestamp column is datetime with timezone
        if not pd.api.types.is_datetime64_dtype(df_copy['fundingRateTimestamp']):
            df_copy['fundingRateTimestamp'] = pd.to_datetime(df_copy['fundingRateTimestamp'])
        
        # Add timezone if missing
        if df_copy['fundingRateTimestamp'].dt.tz is None:
            df_copy['fundingRateTimestamp'] = df_copy['fundingRateTimestamp'].dt.tz_localize(pytz.UTC)
        
        # Filter by date range if provided
        if start_date:
            df_copy = df_copy[df_copy['fundingRateTimestamp'] >= start_date]
        if end_date:
            df_copy = df_copy[df_copy['fundingRateTimestamp'] <= end_date]
        
        if df_copy.empty:
            continue
        
        # Ensure fundingRate is numeric
        df_copy['fundingRate'] = pd.to_numeric(df_copy['fundingRate'], errors='coerce')
        
        # Calculate absolute funding rate
        df_copy['abs_funding_rate'] = df_copy['fundingRate'].abs()
        
        # Get funding interval (in minutes)
        funding_interval = df_copy['fundingInterval'].iloc[0]
        
        # Normalize funding rate if requested
        if normalized:
            # Calculate normalization factor (how many funding events per day)
            normalization_factor = 1440 / funding_interval  # 1440 minutes in a day
            
            # Normalize the funding rate to a daily equivalent rate
            df_copy['normalized_abs_rate'] = df_copy['abs_funding_rate'] * normalization_factor
        else:
            # Just use the raw rate
            df_copy['normalized_abs_rate'] = df_copy['abs_funding_rate']
        
        # Add symbol column
        df_copy['symbol'] = symbol
        
        # Extract date (drop time part) for daily aggregation
        df_copy['date'] = df_copy['fundingRateTimestamp'].dt.date
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        
        # Append to all data
        all_funding_data.append(df_copy)
    
    # Check if we have any data
    if not all_funding_data:
        print("No data found for the specified period")
        return None
    
    # Combine all data
    combined_df = pd.concat(all_funding_data, ignore_index=True)
    
    # Aggregate by date according to specified metric
    if metric == "max":
        daily_agg = combined_df.groupby('date')['normalized_abs_rate'].max().reset_index()
        metric_label = "Maximum"
    else:  # default to mean
        daily_agg = combined_df.groupby('date')['normalized_abs_rate'].mean().reset_index()
        metric_label = "Mean"
    
    # Also calculate count of symbols per day for reference
    symbol_count = combined_df.groupby('date')['symbol'].nunique().reset_index()
    symbol_count.rename(columns={'symbol': 'symbol_count'}, inplace=True)
    
    # Merge with the main DataFrame
    daily_agg = pd.merge(daily_agg, symbol_count, on='date')
    
    # Create the time series plot
    plt.figure(figsize=(15, 8))
    
    # Plot the main line
    plt.plot(daily_agg['date'], daily_agg['normalized_abs_rate'], 
             color='black', linewidth=1.5)
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Format x-axis as dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Show every 3 months
    
    # Additional formatting
    plt.gcf().autofmt_xdate()  # Auto-format date labels
    
    # Add labels and title
    normalized_label = "Normalized (Daily Equivalent)" if normalized else "Raw"
    plt.title(f'{metric_label} Absolute Perpetual Funding Rate {normalized_label}', fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel(f'{metric_label} Absolute Funding Rate', fontsize=14)
    
    # Add symbol count as text in the corner
    avg_symbols = daily_agg['symbol_count'].mean()
    max_symbols = daily_agg['symbol_count'].max()
    plt.figtext(0.01, 0.01, 
                f"Avg. Symbols per Day: {avg_symbols:.1f}\nMax Symbols: {max_symbols:.0f}",
                ha='left', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    # Add more context
    date_range_text = (f"Date Range: {daily_agg['date'].min().strftime('%Y-%m-%d')} to "
                      f"{daily_agg['date'].max().strftime('%Y-%m-%d')}")
    
    # Identify interesting peaks for annotation
    threshold = daily_agg['normalized_abs_rate'].mean() + 2 * daily_agg['normalized_abs_rate'].std()
    peaks = daily_agg[daily_agg['normalized_abs_rate'] > threshold].copy()
    
    # Limit to at most 5 annotations to avoid clutter
    if len(peaks) > 5:
        peaks = peaks.sort_values('normalized_abs_rate', ascending=False).head(5)
    
    # Annotate peaks
    for _, peak in peaks.iterrows():
        plt.annotate(
            f"{peak['date'].strftime('%Y-%m-%d')}\n{peak['normalized_abs_rate']:.3f}",
            xy=(peak['date'], peak['normalized_abs_rate']),
            xytext=(10, 20),  # Offset text
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7),
            fontsize=8
        )
    
    # Add statistical information
    stats_text = (
        f"Statistics:\n"
        f"Mean: {daily_agg['normalized_abs_rate'].mean():.4f}\n"
        f"Median: {daily_agg['normalized_abs_rate'].median():.4f}\n"
        f"Max: {daily_agg['normalized_abs_rate'].max():.4f}\n"
        f"Min: {daily_agg['normalized_abs_rate'].min():.4f}\n"
        f"{date_range_text}"
    )
    
    plt.annotate(
        stats_text, 
        xy=(0.01, 0.99),  # Top left position
        xycoords='axes fraction',
        va='top',  # Vertical alignment
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7),
        fontsize=10
    )
    
    # Save the figure
    metric_name = "max" if metric == "max" else "mean"
    norm_label = "normalized" if normalized else "raw"
    plt.tight_layout()
    output_path = f"{output_dir}/{metric_name}_abs_funding_rate_{norm_label}.png"
    plt.savefig(output_path, dpi=300)
    print(f"Time series plot saved as {output_path}")
    
    # Show the plot
    plt.show()
    
    # Return the data for further analysis
    return daily_agg

def plot_funding_volatility(funding_data, output_dir="funding_analysis", 
                           start_date=None, end_date=None, window=7):
    """
    Plot the volatility of funding rates over time to identify periods of high opportunity.
    
    Args:
        funding_data (dict): Dictionary with symbols as keys and funding rate DataFrames as values
        output_dir (str): Directory to save results
        start_date (str, optional): Start date in 'YYYY-MM-DD' format
        end_date (str, optional): End date in 'YYYY-MM-DD' format
        window (int): Rolling window size for volatility calculation
        
    Returns:
        pd.DataFrame: DataFrame with volatility metrics
    """
    # First get the daily aggregated rates
    daily_rates = plot_aggregated_funding_rates(
        funding_data, output_dir=output_dir, 
        start_date=start_date, end_date=end_date, 
        metric="mean", normalized=True
    )
    
    if daily_rates is None or len(daily_rates) < window:
        print(f"Not enough data for volatility analysis (need at least {window} days)")
        return None
    
    # Calculate rolling statistics
    daily_rates['rolling_mean'] = daily_rates['normalized_abs_rate'].rolling(window=window).mean()
    daily_rates['rolling_std'] = daily_rates['normalized_abs_rate'].rolling(window=window).std()
    daily_rates['rolling_cv'] = daily_rates['rolling_std'] / daily_rates['rolling_mean']  # Coefficient of variation
    
    # Calculate rolling Sharpe ratio-like metric (mean/std)
    daily_rates['rolling_sharpe'] = daily_rates['rolling_mean'] / daily_rates['rolling_std']
    
    # Drop rows with NaN from rolling calculations
    daily_rates = daily_rates.dropna()
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    
    # Create subplots
    ax1 = plt.subplot(211)  # Top subplot for rates
    ax2 = plt.subplot(212, sharex=ax1)  # Bottom subplot for volatility metrics
    
    # Plot rates on top subplot
    ax1.plot(daily_rates['date'], daily_rates['normalized_abs_rate'], 
            color='black', linewidth=1, alpha=0.6, label='Daily Rate')
    ax1.plot(daily_rates['date'], daily_rates['rolling_mean'], 
            color='blue', linewidth=2, label=f'{window}-Day Moving Avg')
    
    # Add bands for standard deviation
    ax1.fill_between(
        daily_rates['date'],
        daily_rates['rolling_mean'] - daily_rates['rolling_std'],
        daily_rates['rolling_mean'] + daily_rates['rolling_std'],
        color='blue', alpha=0.2, label=f'±1 Std Dev'
    )
    
    # Plot volatility metrics on bottom subplot
    ax2.plot(daily_rates['date'], daily_rates['rolling_cv'], 
            color='purple', linewidth=2, label='Coefficient of Variation')
    ax2.plot(daily_rates['date'], daily_rates['rolling_sharpe'], 
            color='green', linewidth=2, label='Sharpe Ratio')
    
    # Add horizontal line at y=1 for reference on Sharpe ratio
    ax2.axhline(y=1, color='green', linestyle='--', alpha=0.5)
    
    # Format axes
    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best')
    
    # Format x-axis dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gcf().autofmt_xdate()
    
    # Add labels
    ax1.set_ylabel('Absolute Funding Rate', fontsize=12)
    ax1.set_title(f'Normalized Absolute Funding Rate and {window}-Day Moving Average', fontsize=14)
    
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Metric Value', fontsize=12)
    ax2.set_title('Funding Rate Volatility Metrics', fontsize=14)
    
    # Add annotations for highest opportunity periods (highest Sharpe ratio)
    top_opportunities = daily_rates.sort_values('rolling_sharpe', ascending=False).head(3)
    
    for _, row in top_opportunities.iterrows():
        ax2.annotate(
            f"{row['date'].strftime('%Y-%m-%d')}\nSharpe: {row['rolling_sharpe']:.2f}",
            xy=(row['date'], row['rolling_sharpe']),
            xytext=(10, -30),
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7),
            fontsize=8
        )
    
    # Save the figure
    plt.tight_layout()
    output_path = f"{output_dir}/funding_rate_volatility_w{window}.png"
    plt.savefig(output_path, dpi=300)
    print(f"Volatility analysis plot saved as {output_path}")
    
    # Show the plot
    plt.show()
    
    return daily_rates
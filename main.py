from utility.bybit_quantlib import *


if __name__ == "__main__":
    start_date = "2024-01-01"
    end_date = "2025-05-16"
    
    window=7
    remove_overlapping=True
    output_dir="funding_analysis"
    
    """
    Run complete funding rate persistence analysis
    
    Args:
        symbols_path (str): Path to CSV file with symbols
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        window (int): Window size for moving average
        remove_overlapping (bool): Whether to remove overlapping periods
        output_dir (str): Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Step 1: Loading symbols...")
    symbols_df = load_symbols("symbols.csv")
    print(f"Loaded {len(symbols_df)} symbols")

    
    print(f"Step 2: Fetching funding rate data from {start_date} to {end_date}...")
    funding_data = fetch_all_funding_rates(symbols_df, start_date, end_date, 
                                          output_dir=f"{output_dir}/raw_data")
    
    print(f"Fetched data for {len(funding_data)} symbols")
    
    print("Step 3: Calculating daily mean funding rates...")
    daily_mean = calculate_daily_mean_funding(funding_data)
    
    daily_mean_path = f"{output_dir}/daily_mean_funding.csv"
    daily_mean_save = daily_mean.copy()
    daily_mean_save['date'] = daily_mean_save['date'].dt.strftime('%Y-%m-%d %H:%M:%S%z')
    daily_mean_save.to_csv(daily_mean_path, index=False)
    print(f"Daily mean funding rates saved to {daily_mean_path}")
    
    print(f"Step 4: Calculating {window}-day moving averages and future funding rates...")
    analysis_df = calculate_moving_averages_and_future_funding(
        daily_mean, window, remove_overlapping)
    
    # Save analysis DataFrame
    analysis_path = f"{output_dir}/funding_persistence_analysis.csv"
    # Convert timezone-aware datetime to string before saving to CSV
    analysis_save = analysis_df.copy()
    analysis_save['date'] = analysis_save['date'].dt.strftime('%Y-%m-%d %H:%M:%S%z')
    analysis_save.to_csv(analysis_path, index=False)
    print(f"Analysis data saved to {analysis_path}")
    
    print("Step 5: Creating funding rate persistence plot...")
    plot_funding_rate_persistence(analysis_df, f"{output_dir}/funding_persistence.png")
    
    print("Step 6: Analyzing funding spread by symbol...")
    symbol_stats = analyze_funding_spread_by_symbol(
        funding_data,         
        symbols_df,           
        output_dir=output_dir,
        days_to_analyze=7
        )

    print("Analysis completed!")
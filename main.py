from utility.bybit_quantlib import *

if __name__ == "__main__":
    start_date = "2024-01-01"
    end_date = "2025-05-01" # prima era 2023-01-01 >> 2025-05-16
    
    window=3
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

    '''
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
    plot_funding_rate_persistence(analysis_df, f"{output_dir}/funding_persistence.png", window=window)
    
    print("Step 6: Analyzing funding spread by symbol...")
    symbol_stats = analyze_funding_spread_by_symbol(
        funding_data,         
        symbols_df,           
        output_dir=output_dir,
        days_to_analyze=window
        )

    print("Step 7: Creating aggregated funding rate time series...")
    daily_avg_df = plot_aggregated_funding_rates(
        funding_data, 
        symbols_df, 
        output_dir=output_dir,
        start_date=start_date,
        end_date=end_date,
        metric="mean",   
        normalized=True 
    )

    print("Analysis completed!")

    '''

    import logging
    from utility.bybit_klines import *
    from dotenv import load_dotenv


    interval = "1h"  
    kline_output_dir = "kline_data"
    
    load_dotenv()

    api_key = os.getenv("BYBIT_TEST_API_KEY")
    api_secret = os.getenv("BYBIT_TEST_API_SECRET")
    
    # Crea una sessione Bybit come facevi prima
    session = HTTP(
        testnet=False,
        api_key=api_key,
        api_secret=api_secret
    )
     # Configura logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("main_process.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Step 1: Loading symbols...")
    symbols_df = load_symbols("symbols.csv")
    logger.info(f"Loaded {len(symbols_df)} symbols")
    
    interval = "1h"  # ... "1m", "5m", "15m", "30m", "4h", "1d", ecc. 
    kline_output_dir = "kline_data"
    max_workers = 5

    max_symbols = None
    
    logger.info(f"Step 2: Fetching {interval} kline data from {start_date} to {end_date}...")
    
    download_spot = False
    download_fut = False


    if download_fut == True:
            
        try:
            future_data = fetch_all_futures_parallel(
                session=session,
                symbols_df=symbols_df,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
                output_dir=kline_output_dir,
                max_workers=max_workers,
                max_symbols=max_symbols
            )
            logger.info(f"Fetched future data for {len(future_data)} symbols")
        except Exception as e:
            logger.error(f"Error in fetch_all_futures_parallel: {e}")
            future_data = {}

        
    if download_spot == True:

        try:
            spot_data = fetch_all_spot_parallel(
            session=session,
            symbols_df=symbols_df,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            output_dir=kline_output_dir,
            max_workers=max_workers,
            max_symbols=max_symbols
        )
            logger.info(f"Fetched future data for {len(future_data)} symbols")
        except Exception as e:
            logger.error(f"Error in fetch_all_futures_parallel: {e}")
            future_data = {}

    
    print("\nStep 8: Running funding strategy backtest...")
    
    '''
    results_df, rebalance_history = backtest_funding_strategy(
        funding_data, 
        symbols_df=symbols_df,
        output_dir=output_dir,
        start_date=start_date,
        end_date=end_date,
        initial_capital=1000,
        min_annual_rate=30.0,
        top_n=5,
        rebalance_days=7
    )
    '''
    from strategy import backtest_funding_strategy_with_trading, analyze_data_availability

    results, rebalances, availability = backtest_funding_strategy_with_trading(
        funding_data=funding_data,
        symbols_df=symbols_df,
        start_date=start_date,
        end_date=end_date
    )
    
    # Analyze data availability
    if availability:
        analyze_data_availability(availability)
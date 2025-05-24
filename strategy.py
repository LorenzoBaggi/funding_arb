import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pytz
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from collections import defaultdict
import concurrent.futures

def backtest_funding_strategy(funding_data, symbols_df=None, output_dir="funding_analysis",
                             start_date=None, end_date=None, 
                             initial_capital=1000, min_annual_rate=30.0,
                             top_n=5, rebalance_days=7,
                             # Parametri dei costi
                             taker_fee_pct=0.05,  # 0.05% per trade
                             slippage_pct=0.03,   # 0.03% di slippage medio
                             bid_ask_spread_pct=0.02):  # 0.02% di spread bid-ask medio
    """
    Backtest a funding rate trading strategy that:
    1. Identifies top N coins with highest funding rates above threshold
    2. Invests equally in each coin
    3. Rebalances portfolio at specified intervals
    4. Tracks PnL from collected funding payments
    
    Args:
        funding_data (dict): Dictionary with symbols as keys and funding rate DataFrames as values
        symbols_df (pd.DataFrame, optional): DataFrame with symbols metadata
        output_dir (str): Directory to save results
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        initial_capital (float): Starting capital in USD
        min_annual_rate (float): Minimum annualized funding rate to consider (%)
        top_n (int): Number of top coins to include in portfolio
        rebalance_days (int): Number of days between portfolio rebalancing
        taker_fee_pct (float): Trading fee as percentage
        slippage_pct (float): Estimated slippage as percentage
        bid_ask_spread_pct (float): Estimated bid-ask spread as percentage
        
    Returns:
        pd.DataFrame: DataFrame with backtest results
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting backtest from {start_date} to {end_date} with ${initial_capital} initial capital")
    print(f"Strategy: Investing in top {top_n} coins with >={min_annual_rate}% annualized funding rate")
    print(f"Rebalancing every {rebalance_days} days")
    print(f"Trading costs: Taker fee {taker_fee_pct}%, Slippage {slippage_pct}%, Bid-ask {bid_ask_spread_pct}%")
    
    # Convert date strings to datetime objects
    start_dt = pd.to_datetime(start_date).tz_localize(pytz.UTC)
    end_dt = pd.to_datetime(end_date).tz_localize(pytz.UTC)
    
    # Step 1: Prepare all funding data with normalized rates - PARALLELIZED
    all_funding_data = []
    
    # Define function to process each symbol's data
    def process_symbol_data(symbol_df_tuple):
        symbol, df = symbol_df_tuple
        if df.empty:
            return None
        
        # Make a copy to avoid modifying original data
        df_copy = df.copy()
        
        # Ensure timestamp column is datetime with timezone
        if not pd.api.types.is_datetime64_dtype(df_copy['fundingRateTimestamp']):
            df_copy['fundingRateTimestamp'] = pd.to_datetime(df_copy['fundingRateTimestamp'])
        
        # Add timezone if missing
        if df_copy['fundingRateTimestamp'].dt.tz is None:
            df_copy['fundingRateTimestamp'] = df_copy['fundingRateTimestamp'].dt.tz_localize(pytz.UTC)
        
        # Filter by date range
        df_copy = df_copy[(df_copy['fundingRateTimestamp'] >= start_dt) & 
                          (df_copy['fundingRateTimestamp'] <= end_dt)]
        
        if df_copy.empty:
            return None
        
        # Ensure fundingRate is numeric
        df_copy['fundingRate'] = pd.to_numeric(df_copy['fundingRate'], errors='coerce')
        
        # Get funding interval (in minutes)
        funding_interval_min = df_copy['fundingInterval'].iloc[0]
        
        # Calculate normalization factor for annualization
        funding_events_per_year = 365 * 24 * 60 / funding_interval_min
        
        # Calculate annualized funding rate - vectorized operation
        df_copy['annualized_rate'] = df_copy['fundingRate'] * funding_events_per_year * 100
        
        # Add symbol column
        df_copy['symbol'] = symbol
        df_copy['funding_interval_min'] = funding_interval_min
        
        return df_copy
    
    # Use ThreadPoolExecutor for I/O-bound tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, os.cpu_count()*4)) as executor:
        # Submit all tasks
        future_to_symbol = {executor.submit(process_symbol_data, (symbol, df)): symbol 
                            for symbol, df in funding_data.items()}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_symbol):
            result = future.result()
            if result is not None:
                all_funding_data.append(result)
    
    if not all_funding_data:
        print("No data found for the specified period")
        return None, None
    
    # Combine all data
    combined_df = pd.concat(all_funding_data, ignore_index=True)
    combined_df = combined_df.sort_values('fundingRateTimestamp')
    
    # Step 2: Create a date range with hourly frequency for our results dataframe
    hours_range = pd.date_range(start=start_dt, end=end_dt, freq='H')
    results_df = pd.DataFrame(index=hours_range)
    results_df.index.name = 'timestamp'
    results_df = results_df.reset_index()
    
    # Initialize columns for tracking
    results_df['portfolio_value'] = initial_capital
    results_df['funding_pnl'] = 0.0
    results_df['trading_pnl'] = 0.0  # To track trading PnL from buy low/sell high
    results_df['trading_costs'] = 0.0  # To track trading costs
    results_df['period_return'] = 0.0
    results_df['cumulative_return'] = 0.0
    
    # Step 3: Determine rebalance dates (every X days starting from start_date)
    rebalance_dates = pd.date_range(start=start_dt, end=end_dt, freq=f'{rebalance_days}D')
    
    # Dictionary to track current portfolio holdings
    portfolio = {}
    previous_portfolio = {}
    
    # Lists to store trading cost details
    trading_costs = []
    
    # Dictionary to store all rebalance events for later visualization and analysis
    rebalance_history = []
    
    # Step 4: Perform the backtest
    print("Running backtest...")
    
    # Find the index of the first hour in our results dataframe
    current_idx = 0
    current_capital = initial_capital
    
    for i, rebalance_date in enumerate(rebalance_dates):
        # Determine end date for this period (next rebalance date or end of backtest)
        if i < len(rebalance_dates) - 1:
            next_rebalance = rebalance_dates[i+1]
        else:
            next_rebalance = end_dt
            
        print(f"Rebalance period {i+1}: {rebalance_date.strftime('%Y-%m-%d')} to {next_rebalance.strftime('%Y-%m-%d')}")
        
        # Get data for the next 7 days to evaluate expected performance
        forecast_start = rebalance_date
        forecast_end = rebalance_date + timedelta(days=rebalance_days)
        
        # Get all symbols active during this forecast period - use more efficient filtering
        forecast_data = combined_df[
            (combined_df['fundingRateTimestamp'] >= forecast_start) & 
            (combined_df['fundingRateTimestamp'] < forecast_end)
        ]
        
        # Group by symbol and calculate average annualized rate for the forecast period
        forecast_rates = forecast_data.groupby('symbol', as_index=False).agg({
            'annualized_rate': 'mean',
            'funding_interval_min': 'first'
        })
        
        # Filter by minimum annual rate threshold
        forecast_rates = forecast_rates[forecast_rates['annualized_rate'] >= min_annual_rate]
        
        # Sort by annualized rate (descending) and get top N symbols
        if not forecast_rates.empty:
            forecast_rates = forecast_rates.sort_values('annualized_rate', ascending=False)
            top_symbols = forecast_rates.head(top_n)
            
            # Record symbols that are entering the portfolio
            new_symbols = set(top_symbols['symbol']) - set(portfolio.keys())
            
            # Record symbols that are leaving the portfolio
            removed_symbols = set(portfolio.keys()) - set(top_symbols['symbol'])
            
            # Calculate allocation per symbol (equal weight)
            allocation_per_symbol = current_capital / min(top_n, len(top_symbols))
            
            # Calculate trading costs if not the first rebalance
            total_trading_cost = 0
            
            if i > 0:  # Skip first rebalance as it's the initial portfolio setup
                # Calculate costs for exiting positions
                for symbol in removed_symbols:
                    exit_amount = previous_portfolio[symbol]
                    
                    # Calculate trading costs
                    taker_fee = exit_amount * (taker_fee_pct / 100)
                    slippage = exit_amount * (slippage_pct / 100)
                    bid_ask = exit_amount * (bid_ask_spread_pct / 100) / 2  # Half the spread
                    
                    total_cost = taker_fee + slippage + bid_ask
                    total_trading_cost += total_cost
                    
                    # Record trading costs
                    trading_costs.append({
                        'date': rebalance_date,
                        'symbol': symbol,
                        'action': 'exit',
                        'amount': exit_amount,
                        'taker_fee': taker_fee,
                        'slippage': slippage,
                        'bid_ask': bid_ask,
                        'total_cost': total_cost
                    })
                
                # Calculate costs for adjusting existing positions
                for symbol in set(portfolio.keys()) & set(top_symbols['symbol']):
                    old_allocation = previous_portfolio[symbol]
                    new_allocation = allocation_per_symbol
                    
                    # If allocation changes, calculate rebalancing costs
                    if old_allocation != new_allocation:
                        rebalance_amount = abs(new_allocation - old_allocation)
                        
                        # Calculate trading costs
                        taker_fee = rebalance_amount * (taker_fee_pct / 100)
                        slippage = rebalance_amount * (slippage_pct / 100)
                        bid_ask = rebalance_amount * (bid_ask_spread_pct / 100) / 2
                        
                        total_cost = taker_fee + slippage + bid_ask
                        total_trading_cost += total_cost
                        
                        # Record trading costs
                        trading_costs.append({
                            'date': rebalance_date,
                            'symbol': symbol,
                            'action': 'rebalance',
                            'amount': rebalance_amount,
                            'taker_fee': taker_fee,
                            'slippage': slippage,
                            'bid_ask': bid_ask,
                            'total_cost': total_cost
                        })
                
                # Calculate costs for new entries
                for symbol in new_symbols:
                    entry_amount = allocation_per_symbol
                    
                    # Calculate trading costs
                    taker_fee = entry_amount * (taker_fee_pct / 100)
                    slippage = entry_amount * (slippage_pct / 100)
                    bid_ask = entry_amount * (bid_ask_spread_pct / 100) / 2
                    
                    total_cost = taker_fee + slippage + bid_ask
                    total_trading_cost += total_cost
                    
                    # Record trading costs
                    trading_costs.append({
                        'date': rebalance_date,
                        'symbol': symbol,
                        'action': 'entry',
                        'amount': entry_amount,
                        'taker_fee': taker_fee,
                        'slippage': slippage,
                        'bid_ask': bid_ask,
                        'total_cost': total_cost
                    })
                
                # Deduct trading costs from capital
                current_capital -= total_trading_cost
                
                # Find corresponding hour index in results_df
                hour_idx = results_df[results_df['timestamp'] == rebalance_date].index
                if len(hour_idx) > 0:
                    results_df.loc[hour_idx[0], 'trading_costs'] = -total_trading_cost
            
            # Update portfolio with new allocations after costs
            portfolio = {}
            if current_capital > 0:
                # Recalculate allocation after costs
                allocation_per_symbol = current_capital / min(top_n, len(top_symbols))
                portfolio = {symbol: allocation_per_symbol for symbol in top_symbols['symbol']}
            
            # Store rebalance event
            rebalance_event = {
                'date': rebalance_date,
                'portfolio': portfolio.copy(),
                'portfolio_value': current_capital,
                'top_symbols': top_symbols.to_dict('records'),
                'new_symbols': list(new_symbols),
                'removed_symbols': list(removed_symbols),
                'trading_costs': total_trading_cost if i > 0 else 0
            }
            rebalance_history.append(rebalance_event)
            
            # Save previous portfolio for next rebalance
            previous_portfolio = portfolio.copy()
            
            print(f"Selected {len(portfolio)} symbols with allocation ${allocation_per_symbol:.2f} each")
            for _, row in top_symbols.iterrows():
                print(f"  {row['symbol']}: {row['annualized_rate']:.2f}% annualized rate")
                
            if i > 0:
                print(f"Trading costs: ${total_trading_cost:.2f}")
        else:
            print("No symbols meet the minimum funding rate criteria for this period")
            portfolio = {}
            rebalance_event = {
                'date': rebalance_date,
                'portfolio': {},
                'portfolio_value': current_capital,
                'top_symbols': [],
                'new_symbols': [],
                'removed_symbols': list(portfolio.keys()),
                'trading_costs': 0
            }
            rebalance_history.append(rebalance_event)
            previous_portfolio = portfolio.copy()
        
        # Process each hour in this period - use vectorized operations where possible
        period_hours = results_df[
            (results_df['timestamp'] >= rebalance_date) & 
            (results_df['timestamp'] < next_rebalance)
        ].index.tolist()
        
        for hour_idx in period_hours:
            hour_timestamp = results_df.loc[hour_idx, 'timestamp']
            
            # Calculate funding received for this hour
            hour_funding_pnl = 0.0
            
            # Check each symbol in portfolio - this loop is hard to vectorize due to time matching
            for symbol, allocation in portfolio.items():
                # Find if there's a funding payment at this hour
                symbol_funding = combined_df[
                    (combined_df['symbol'] == symbol) & 
                    (combined_df['fundingRateTimestamp'] == hour_timestamp)
                ]
                
                if not symbol_funding.empty:
                    # Calculate funding payment
                    funding_rate = symbol_funding['fundingRate'].iloc[0]
                    funding_payment = allocation * funding_rate
                    hour_funding_pnl += funding_payment
            
            # Update results for this hour
            results_df.loc[hour_idx, 'funding_pnl'] = hour_funding_pnl
            
            # Update portfolio value
            if hour_idx > 0:
                previous_value = results_df.loc[hour_idx-1, 'portfolio_value']
                # Include trading costs (already negative)
                hour_costs = results_df.loc[hour_idx, 'trading_costs']
                results_df.loc[hour_idx, 'portfolio_value'] = previous_value + hour_funding_pnl + hour_costs
            else:
                previous_value = initial_capital
                hour_costs = results_df.loc[hour_idx, 'trading_costs']
                results_df.loc[hour_idx, 'portfolio_value'] = previous_value + hour_funding_pnl + hour_costs
            
            # Update current capital for next rebalance
            current_capital = results_df.loc[hour_idx, 'portfolio_value']
    
    # Step 5: Calculate period returns and cumulative returns
    # Use vectorized operations for efficiency
    results_df['period_return'] = results_df['portfolio_value'].pct_change()
    results_df.loc[0, 'period_return'] = (results_df.loc[0, 'portfolio_value'] - initial_capital) / initial_capital
    
    # Replace NaN and inf values with 0
    results_df['period_return'] = results_df['period_return'].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Calculate cumulative return - vectorized
    results_df['cumulative_return'] = (1 + results_df['period_return']).cumprod() - 1
    
    # Fill in the trading_pnl column (if we had price data, we'd calculate actual trading PnL)
    # For now, we'll just add a placeholder value
    results_df['trading_pnl'] = 0.0
    
    # Calculate cumulative costs and pnl metrics for the breakdown chart
    results_df['cum_funding_pnl'] = results_df['funding_pnl'].cumsum()
    results_df['cum_trading_costs'] = results_df['trading_costs'].cumsum()
    results_df['cum_trading_pnl'] = results_df['trading_pnl'].cumsum()
    results_df['total_pnl'] = results_df['cum_funding_pnl'] + results_df['cum_trading_pnl'] + results_df['cum_trading_costs']
    
    # Calculate additional performance metrics
    total_days = (end_dt - start_dt).days
    total_return = results_df['portfolio_value'].iloc[-1] / initial_capital - 1
    annual_return = ((1 + total_return) ** (365 / total_days)) - 1
    
    # Resample daily for efficiency in calculating volatility
    daily_returns = results_df.set_index('timestamp').resample('D')['period_return'].sum()
    volatility = daily_returns.std() * np.sqrt(365)  # Annualized
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    
    # Calculate drawdowns
    results_df['peak'] = results_df['portfolio_value'].cummax()
    results_df['drawdown'] = (results_df['portfolio_value'] - results_df['peak']) / results_df['peak']
    max_drawdown = results_df['drawdown'].min()
    
    # Summarize trading costs
    if trading_costs:
        costs_df = pd.DataFrame(trading_costs)
        total_costs = costs_df['total_cost'].sum()
        costs_by_type = costs_df.groupby('action')['total_cost'].sum()
        
        print("\nTrading Costs Summary:")
        print(f"Total Trading Costs: ${total_costs:.2f}")
        print(costs_by_type)
        
        # Calculate impact on returns
        cost_impact_pct = total_costs / initial_capital * 100
        print(f"Cost Impact: {cost_impact_pct:.2f}% of initial capital")
    else:
        total_costs = 0
    
    # Print performance summary
    print("\nPerformance Summary:")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return: {annual_return:.2%}")
    print(f"Annualized Volatility: {volatility:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"Final Portfolio Value: ${results_df['portfolio_value'].iloc[-1]:.2f}")
    
    # Step 6: Plot results
    plot_backtest_results(results_df, rebalance_history, output_dir, initial_capital)
    
    # Plot PnL breakdown
    plot_pnl_breakdown(results_df, output_dir, initial_capital)
    
    # Save results dataframe
    results_path = f"{output_dir}/funding_strategy_backtest_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Backtest results saved to {results_path}")
    
    # Save rebalance history
    if rebalance_history:
        # Convert portfolio dictionaries to strings for CSV storage
        for event in rebalance_history:
            event['portfolio'] = str(event['portfolio'])
            if 'top_symbols' in event:
                event['top_symbols'] = str(event['top_symbols'])
        
        rebalance_path = f"{output_dir}/funding_strategy_rebalance_history.csv"
        rebalance_df = pd.DataFrame(rebalance_history)
        rebalance_df.to_csv(rebalance_path, index=False)
        print(f"Rebalance history saved to {rebalance_path}")
    
    return results_df, rebalance_history

def plot_pnl_breakdown(results_df, output_dir, initial_capital):
    """
    Plot a breakdown of PnL components: total PnL, funding PnL, trading PnL, and costs
    
    Args:
        results_df (pd.DataFrame): DataFrame with backtest results
        output_dir (str): Directory to save results
        initial_capital (float): Initial capital amount
    """
    # Create figure with 4 subplots stacked vertically
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    
    # Use daily resampling for cleaner visualization
    daily_df = results_df.set_index('timestamp').resample('D').last().reset_index()
    
    # Plot 1: Total PnL
    axes[0].plot(daily_df['timestamp'], daily_df['total_pnl'], 'r-', linewidth=1.5)
    axes[0].set_title('Total PnL', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.2)
    
    # Plot 2: Funding PnL
    axes[1].plot(daily_df['timestamp'], daily_df['cum_funding_pnl'], 'g-', linewidth=1.5)
    axes[1].set_title('Funding PnL', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.2)
    
    # Plot 3: Trading PnL
    axes[2].plot(daily_df['timestamp'], daily_df['cum_trading_pnl'], 'b-', linewidth=1.5)
    axes[2].set_title('Trading PnL', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.2)
    
    # Plot 4: Costs
    axes[3].plot(daily_df['timestamp'], daily_df['cum_trading_costs'], 'purple', linewidth=1.5)
    axes[3].set_title('Costs', fontsize=12)
    axes[3].grid(True, alpha=0.3)
    axes[3].axhline(y=0, color='black', linestyle='-', alpha=0.2)
    
    # Formatting
    for ax in axes:
        ax.set_ylabel('PnL', fontsize=10)
    
    axes[3].set_xlabel('Date', fontsize=12)
    
    # Format dates on x-axis
    plt.gcf().autofmt_xdate()
    
    # Add title for the entire figure
    plt.suptitle(f'PnL to price change, funding, and costs. Starting capital ${initial_capital}', 
                fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save figure
    output_path = f"{output_dir}/pnl_breakdown.png"
    plt.savefig(output_path, dpi=300)
    print(f"PnL breakdown chart saved to {output_path}")
    
    plt.show()

def plot_backtest_results(results_df, rebalance_history, output_dir, initial_capital):
    """
    Plot the results of the funding strategy backtest
    
    Args:
        results_df (pd.DataFrame): DataFrame with backtest results
        rebalance_history (list): List of rebalance events
        output_dir (str): Directory to save results
        initial_capital (float): Initial capital amount
    """
    # Use daily resampling for cleaner visualization
    daily_df = results_df.set_index('timestamp').resample('D').last().reset_index()
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 12))
    
    # Plot 1: Portfolio Value
    ax1 = plt.subplot(311)
    ax1.plot(daily_df['timestamp'], daily_df['portfolio_value'], 'b-', linewidth=1.5)
    
    # Add horizontal line for initial capital
    ax1.axhline(y=initial_capital, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
    
    # Add vertical lines for rebalance dates
    for event in rebalance_history:
        ax1.axvline(x=event['date'], color='g', linestyle=':', alpha=0.5)
    
    # Mark specific portfolio events
    for i, event in enumerate(rebalance_history):
        # Add an annotation for major rebalances (e.g., every 5th rebalance)
        if i % 5 == 0 and isinstance(event['portfolio'], dict) and event['portfolio']:
            symbols_text = ", ".join(list(event['portfolio'].keys())[:3])
            if len(event['portfolio']) > 3:
                symbols_text += f"... (+{len(event['portfolio'])-3})"
                
            ax1.annotate(f"{symbols_text}",
                xy=(event['date'], event['portfolio_value']),
                xytext=(0, 20),
                textcoords='offset points',
                fontsize=8,
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'),
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)
            )
    
    ax1.set_title('Funding Strategy Portfolio Value Over Time', fontsize=14)
    ax1.set_ylabel('Portfolio Value (USD)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    
    # Plot 2: Cumulative Return
    ax2 = plt.subplot(312, sharex=ax1)
    ax2.plot(daily_df['timestamp'], daily_df['cumulative_return'] * 100, 'g-', linewidth=1.5)
    
    # Add vertical lines for rebalance dates
    for event in rebalance_history:
        ax2.axvline(x=event['date'], color='g', linestyle=':', alpha=0.3)
    
    ax2.set_title('Cumulative Return (%)', fontsize=14)
    ax2.set_ylabel('Return (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Drawdowns
    ax3 = plt.subplot(313, sharex=ax1)
    ax3.fill_between(daily_df['timestamp'], daily_df['drawdown'] * 100, 0, color='r', alpha=0.3)
    
    # Add vertical lines for rebalance dates
    for event in rebalance_history:
        ax3.axvline(x=event['date'], color='g', linestyle=':', alpha=0.3)
    
    ax3.set_title('Drawdowns (%)', fontsize=14)
    ax3.set_ylabel('Drawdown (%)', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Invert y-axis for drawdowns (so down is negative)
    ax3.invert_yaxis()
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    
    # Calculate performance metrics
    total_days = (daily_df['timestamp'].iloc[-1] - daily_df['timestamp'].iloc[0]).total_seconds() / (60*60*24)
    total_return = daily_df['portfolio_value'].iloc[-1] / initial_capital - 1
    annual_return = ((1 + total_return) ** (365 / total_days)) - 1
    
    # Calculate daily returns for volatility
    daily_returns = results_df.set_index('timestamp').resample('D')['period_return'].sum()
    volatility = daily_returns.std() * np.sqrt(365)  # Annualized
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0
    max_drawdown = daily_df['drawdown'].min()
    
    # Calculate total costs
    total_costs = abs(daily_df['cum_trading_costs'].iloc[-1])

    # Add text with performance metrics
    metrics_text = (
        f"Performance Metrics:\n"
        f"Total Return: {total_return:.2%}\n"
        f"Annualized Return: {annual_return:.2%}\n"
        f"Annualized Volatility: {volatility:.2%}\n"
        f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
        f"Maximum Drawdown: {max_drawdown:.2%}\n"
        f"Initial Capital: ${initial_capital:.2f}\n"
        f"Final Value: ${daily_df['portfolio_value'].iloc[-1]:.2f}\n"
        f"Total Trading Costs: ${total_costs:.2f}\n"
        f"Cost Impact: {(total_costs/initial_capital*100):.2f}% of initial capital"
    )
    
    fig.text(0.02, 0.02, metrics_text, fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save figure
    output_path = f"{output_dir}/funding_strategy_performance.png"
    plt.savefig(output_path, dpi=300)
    print(f"Performance chart saved to {output_path}")
    
    plt.show()
    
    # Create additional chart: Portfolio Composition Over Time
    plt.figure(figsize=(15, 8))
    
    # Prepare data
    dates = [event['date'] for event in rebalance_history]
    
    # Get all unique symbols across all rebalances
    all_symbols = set()
    for event in rebalance_history:
        if isinstance(event['portfolio'], dict):
            all_symbols.update(event['portfolio'].keys())
        elif isinstance(event['portfolio'], str):
            # Handle case where portfolio might have been converted to string
            portfolio_dict = eval(event['portfolio'])
            if isinstance(portfolio_dict, dict):
                all_symbols.update(portfolio_dict.keys())
    
    # Create a dataframe with portfolio weights over time
    portfolio_weights = pd.DataFrame(index=dates, columns=list(all_symbols)).fillna(0)
    
    for i, event in enumerate(rebalance_history):
        portfolio = event['portfolio']
        portfolio_value = event['portfolio_value']
        
        # Handle portfolio as dict or string representation of dict
        if isinstance(portfolio, str):
            try:
                portfolio = eval(portfolio)
            except:
                portfolio = {}
        
        if isinstance(portfolio, dict) and portfolio_value > 0:
            for symbol, allocation in portfolio.items():
                portfolio_weights.loc[event['date'], symbol] = allocation / portfolio_value
    
    # Plot stacked area chart of portfolio weights
    ax = portfolio_weights.plot.area(figsize=(15, 8), stacked=True, alpha=0.7)
    
    # Add labels and title
    plt.title('Portfolio Composition Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Weight', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Improve legend (limit to top 10 symbols by average weight if there are many)
    if len(all_symbols) > 10:
        top_symbols = portfolio_weights.mean().sort_values(ascending=False).head(10).index
        handles, labels = ax.get_legend_handles_labels()
        symbol_indices = [labels.index(symbol) for symbol in top_symbols if symbol in labels]
        selected_handles = [handles[i] for i in symbol_indices]
        selected_labels = [labels[i] for i in symbol_indices]
        plt.legend(selected_handles, selected_labels, loc='center left', bbox_to_anchor=(1, 0.5),
                    title='Top 10 Symbols')
    else:
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Symbols')
    
    plt.tight_layout()
    
    # Save figure
    output_path = f"{output_dir}/funding_strategy_portfolio_composition.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Portfolio composition chart saved to {output_path}")
    
    plt.show()

    # Create a summary of trading costs
    if len(rebalance_history) > 0 and any('trading_costs' in event for event in rebalance_history):
        plt.figure(figsize=(15, 6))
        
        # Extract dates and costs
        rebalance_costs = [(event['date'], event.get('trading_costs', 0)) 
                            for event in rebalance_history if event.get('trading_costs', 0) > 0]
        
        if rebalance_costs:
            dates, costs = zip(*rebalance_costs)
            
            # Plot bar chart of costs
            plt.bar(dates, costs, color='red', alpha=0.7)
            plt.title('Trading Costs at Each Rebalance', fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Cost (USD)', fontsize=12)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Format x-axis dates
            plt.gcf().autofmt_xdate()
            
            # Add cost values as text above bars
            for date, cost in rebalance_costs:
                plt.text(date, cost + max(costs)*0.02, f"${cost:.2f}", 
                            ha='center', va='bottom', fontsize=8, rotation=90)
            
            # Save figure
            output_path = f"{output_dir}/trading_costs_by_rebalance.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            print(f"Trading costs chart saved to {output_path}")
            
            plt.show()
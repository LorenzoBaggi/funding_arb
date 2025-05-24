import pandas as pd


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
    
    # Extract base coin
    if 'baseCoin' in mapping_df.columns and 'quoteCoin' in mapping_df.columns:
        # Check if generated spot symbol is consistent with baseCoin and quoteCoin
        for idx, row in mapping_df.iterrows():
            base = row['baseCoin']
            quote = row['quoteCoin']
            expected_symbol = f"{base}{quote}"
            
            # If baseCoin+quoteCoin symbol is different from generated symbol,
            # use the one based on baseCoin and quoteCoin
            if expected_symbol != row['spotSymbol']:
                logger.info(f"Symbol mismatch: generated {row['spotSymbol']} but using {expected_symbol} based on baseCoin+quoteCoin")
                mapping_df.at[idx, 'spotSymbol'] = expected_symbol
    
    # Save the mapping for debugging
    mapping_path = "future_to_spot_mapping_debug.csv"
    mapping_df[['symbol', 'spotSymbol']].to_csv(mapping_path, index=False)
    
    return mapping_df


if __name__ == "__main__":

    # Test symbols
    test_data = {
        'symbol': [
            '1000000BABYDOGEUSDT',
            '1000PEPEUSDT',
            '10000SATSUSDT',
            '100CATSUSDT',
            'BTCUSDT',
            'ETHUSDT'
        ],
        'baseCoin': [
            '1000000BABYDOGE',
            '1000PEPE',
            '10000SATS',
            '100CATS',
            'BTC',
            'ETH'
        ],
        'quoteCoin': [
            'USDT',
            'USDT',
            'USDT',
            'USDT',
            'USDT',
            'USDT'
        ]
    }

    # Create test DataFrame
    test_df = pd.DataFrame(test_data)

    # Test the mapping function
    result_df = map_future_to_spot(test_df)

    # Print results
    print("Symbol Mapping Results:")
    for _, row in result_df.iterrows():
        print(f"{row['symbol']} -> {row['spotSymbol']}")


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

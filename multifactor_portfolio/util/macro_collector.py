import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict

def update_deribit_dvol(data_dir: str, start_date: str = '2020-01-01') -> pd.DataFrame:
    """
    Downloads and updates Deribit DVOL (Implied Volatility Index) for BTC and ETH using pagination.
    """
    file_path = os.path.join(data_dir, "macro_deribit_dvol.csv.gz")
    existing_df = pd.DataFrame()
    
    if os.path.exists(file_path):
        try:
            existing_df = pd.read_csv(file_path, compression='gzip', index_col=0, parse_dates=True)
        except Exception as e:
            print(f"Error reading DVOL cache: {e}")

    # Determine start timestamp
    if not existing_df.empty:
        # Fetch starting from the last date in cache
        last_date = existing_df.index.max()
        start_ts = int(pd.to_datetime(last_date).timestamp() * 1000)
    else:
        start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
        
    end_ts = int(datetime.now().timestamp() * 1000)
    
    # Only download if there's new data needed
    if existing_df.empty or (datetime.now() - existing_df.index.max()).days >= 1:
        dvol_data = {}
        for cur in ['BTC', 'ETH']:
            cur_data = []
            temp_start = start_ts
            while temp_start < end_ts:
                # Request in chunks of 500 days to be safe (~43,200,000,000 ms)
                temp_end = min(temp_start + 43200000000, end_ts)
                url = f"https://www.deribit.com/api/v2/public/get_volatility_index_data?currency={cur}&resolution=1D&start_timestamp={temp_start}&end_timestamp={temp_end}"
                try:
                    res = requests.get(url, timeout=15)
                    if res.status_code == 200:
                        data = res.json().get('result', {}).get('data', [])
                        if data:
                            cur_data.extend(data)
                            # Advance start to the last date returned + 1 day
                            temp_start = data[-1][0] + 86400000
                        else:
                            break
                    else:
                        break
                except Exception as e:
                    print(f"Failed to fetch Deribit DVOL for {cur}: {e}")
                    break
            
            if cur_data:
                dates = pd.to_datetime([x[0] for x in cur_data], unit='ms').floor('1D')
                closes = [x[4] for x in cur_data]
                dvol_data[f'dvol_{cur.lower()}'] = pd.Series(closes, index=dates)
                
        if dvol_data:
            new_df = pd.DataFrame(dvol_data)
            if not existing_df.empty:
                combined = pd.concat([existing_df, new_df], axis=0)
                combined = combined[~combined.index.duplicated(keep='last')].sort_index()
                combined.to_csv(file_path, compression='gzip')
                return combined
            else:
                new_df.to_csv(file_path, compression='gzip')
                return new_df
                
    return existing_df

def update_fear_greed(data_dir: str) -> pd.DataFrame:
    """
    Downloads and updates Crypto Fear & Greed Index from alternative.me.
    """
    file_path = os.path.join(data_dir, "macro_fear_greed.csv.gz")
    existing_df = pd.DataFrame()
    
    if os.path.exists(file_path):
        try:
            existing_df = pd.read_csv(file_path, compression='gzip', index_col=0, parse_dates=True)
        except Exception as e:
            print(f"Error reading Fear & Greed cache: {e}")
            
    # Always fetch all and merge to ensure completeness
    if existing_df.empty or (datetime.now() - existing_df.index.max()).days >= 1:
        try:
            url = "https://api.alternative.me/fng/?limit=0&format=json"
            res = requests.get(url, timeout=15)
            if res.status_code == 200:
                data = res.json().get('data', [])
                if data:
                    dates = pd.to_datetime([int(x['timestamp']) for x in data], unit='s').floor('1D')
                    vals = [float(x['value']) for x in data]
                    new_df = pd.DataFrame({'fear_greed': vals}, index=dates)
                    
                    if not existing_df.empty:
                        combined = pd.concat([existing_df, new_df], axis=0)
                        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
                        combined.to_csv(file_path, compression='gzip')
                        return combined
                    else:
                        new_df = new_df.sort_index()
                        new_df.to_csv(file_path, compression='gzip')
                        return new_df
        except Exception as e:
            print(f"Failed to fetch Fear & Greed Index: {e}")
            
    return existing_df

def update_defillama_stablecoin(data_dir: str) -> pd.DataFrame:
    """
    Downloads and updates DefiLlama Stablecoin Market Cap.
    """
    file_path = os.path.join(data_dir, "macro_defillama_stablecoin.csv.gz")
    existing_df = pd.DataFrame()
    
    if os.path.exists(file_path):
        try:
            existing_df = pd.read_csv(file_path, compression='gzip', index_col=0, parse_dates=True)
        except Exception as e:
            print(f"Error reading Stablecoin cache: {e}")
            
    if existing_df.empty or (datetime.now() - existing_df.index.max()).days >= 1:
        try:
            url = "https://stablecoins.llama.fi/stablecoincharts/all"
            res = requests.get(url, timeout=15)
            if res.status_code == 200:
                data = res.json()
                if data:
                    dates = pd.to_datetime([int(x['date']) for x in data], unit='s').floor('1D')
                    mcap = [float(x['totalCirculatingUSD']['peggedUSD']) for x in data]
                    new_df = pd.DataFrame({'stablecoin_mcap': mcap}, index=dates)
                    
                    if not existing_df.empty:
                        combined = pd.concat([existing_df, new_df], axis=0)
                        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
                        combined.to_csv(file_path, compression='gzip')
                        return combined
                    else:
                        new_df = new_df.sort_index()
                        new_df.to_csv(file_path, compression='gzip')
                        return new_df
        except Exception as e:
            print(f"Failed to fetch DefiLlama Stablecoin Cap: {e}")
            
    return existing_df

def update_yfinance_macro(data_dir: str, start_date: str = '2020-01-01') -> pd.DataFrame:
    """
    Downloads and updates VIX, DXY, and SPY from yfinance.
    """
    file_path = os.path.join(data_dir, "macro_yfinance.csv.gz")
    existing_df = pd.DataFrame()
    
    if os.path.exists(file_path):
        try:
            existing_df = pd.read_csv(file_path, compression='gzip', index_col=0, parse_dates=True)
        except Exception as e:
            print(f"Error reading yfinance cache: {e}")
            
    if existing_df.empty or (datetime.now() - existing_df.index.max()).days >= 1:
        try:
            import yfinance as yf
            tickers = {'^VIX': 'vix', 'DX-Y.NYB': 'dxy', 'SPY': 'spy'}
            
            # Query from the start date or append
            query_start = start_date
            if not existing_df.empty:
                query_start = existing_df.index.max().strftime('%Y-%m-%d')
                
            data = yf.download(list(tickers.keys()), start=query_start)
            if not data.empty:
                # Clean and extract close prices
                if isinstance(data.columns, pd.MultiIndex):
                    closes = data['Close']
                else:
                    closes = data[['Close']]
                
                new_cols = {}
                for tick, name in tickers.items():
                    if tick in closes.columns:
                        new_cols[name] = closes[tick]
                        
                new_df = pd.DataFrame(new_cols)
                new_df.index = pd.to_datetime(new_df.index)
                
                if not existing_df.empty:
                    combined = pd.concat([existing_df, new_df], axis=0)
                    combined = combined[~combined.index.duplicated(keep='last')].sort_index()
                    combined.to_csv(file_path, compression='gzip')
                    return combined
                else:
                    new_df = new_df.sort_index()
                    new_df.to_csv(file_path, compression='gzip')
                    return new_df
        except Exception as e:
            print(f"Failed to download yfinance macro: {e}")
            
    return existing_df

def download_macro_features(
    data_dir: str, 
    start_date: str = '2020-01-01', 
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Downloads/Updates all macro datasets, joins them on daily frequency,
    and drops features with >20% missing values (NaNs).
    """
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Update / Download all sources
    df_dvol = update_deribit_dvol(data_dir, start_date)
    df_fng = update_fear_greed(data_dir)
    df_stable = update_defillama_stablecoin(data_dir)
    df_yf = update_yfinance_macro(data_dir, start_date)
    
    # 2. Join all DataFrames
    all_dfs = []
    for df in [df_dvol, df_fng, df_stable, df_yf]:
        if not df.empty:
            all_dfs.append(df)
            
    if not all_dfs:
        return pd.DataFrame()
        
    # Join on date index
    macro_df = pd.concat(all_dfs, axis=1)
    
    # Clean index
    macro_df.index = pd.to_datetime(macro_df.index)
    macro_df = macro_df.sort_index()
    
    # Slice to start_date and end_date
    macro_df = macro_df.loc[start_date:]
    if end_date:
        macro_df = macro_df.loc[:end_date]
        
    # Resample to daily frequency and ffill
    macro_df = macro_df.resample('1D').last().ffill()
    
    # 3. Check for completeness: Drop any feature with > 20% NaNs
    final_cols = []
    for col in macro_df.columns:
        nan_pct = macro_df[col].isna().sum() / len(macro_df)
        if nan_pct > 0.2:
            print(f"Warning: Dropping macro feature '{col}' due to {nan_pct * 100:.1f}% missing values.")
        else:
            final_cols.append(col)
            
    macro_df = macro_df[final_cols]
    
    return macro_df

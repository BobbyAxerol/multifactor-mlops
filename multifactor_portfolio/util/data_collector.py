import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from binance import AsyncClient
import nest_asyncio

class FuturesDataCollector:  
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):  
        self.client = None  
        self.api_key = api_key  
        self.api_secret = api_secret  
      
    async def __aenter__(self):  
        self.client = await AsyncClient.create(self.api_key, self.api_secret)  
        return self  
      
    async def __aexit__(self, exc_type, exc_val, exc_tb):  
        if self.client:
            await self.client.close_connection()  
      
    async def get_funding_rate_batch(
        self,
        symbols: List[str],
        start_ts: int,
        end_ts: int
    ) -> pd.DataFrame:
        """
        Fetch full funding rate history (8h) for multiple symbols using time pagination.
        """
        async def fetch_symbol(symbol: str) -> pd.Series:
            all_rows = []
            cur = start_ts
            try:
                while cur < end_ts:
                    data = await self.client.futures_funding_rate(
                        symbol=symbol,
                        startTime=cur,
                        endTime=end_ts,
                        limit=1000
                    )
                    if not data:
                        break
                    all_rows.extend(data)
                    cur = data[-1]["fundingTime"] + 1
                    await asyncio.sleep(0.05)  # soft rate limit

                if not all_rows:
                    return pd.Series(name=symbol, dtype="float64")

                df = pd.DataFrame(all_rows)
                df["fundingTime"] = (
                    pd.to_datetime(df["fundingTime"], unit="ms")
                    .dt.floor("8h")
                )
                df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
                df = (
                    df.drop_duplicates(subset="fundingTime")
                    .set_index("fundingTime")
                    .sort_index()
                )
                return df["fundingRate"].rename(symbol)
            except Exception as e:
                print(f"[Funding] {symbol} error: {e}")
                return pd.Series(name=symbol, dtype="float64")

        tasks = []
        for i, symbol in enumerate(symbols, 1):
            tasks.append(fetch_symbol(symbol))
            if i % 10 == 0:
                await asyncio.sleep(0.1)

        results = await asyncio.gather(*tasks)
        funding_df = pd.concat(results, axis=1).sort_index()
        return funding_df
      
    async def get_oi_ls_ratio_batch(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:  
        """Fetch OI and L/S ratio data for multiple symbols - 4h interval, max 500 records"""  
        async def fetch_oi_data(symbol: str):  
            try:  
                data = await self.client.futures_open_interest_hist(  
                    symbol=symbol, period='4h', limit=500  
                )  
                if data:  
                    df = pd.DataFrame(data)  
                    time_col = 'timestamp' if 'timestamp' in df.columns else 'time'  
                    df[time_col] = pd.to_datetime(df[time_col], unit='ms')  
                    df['openInterest'] = pd.to_numeric(df['sumOpenInterest'], errors='coerce')  
                    return df.set_index(time_col)['openInterest'].rename(f"{symbol}_oi")  
            except Exception as e:  
                print(f"Error fetching OI for {symbol}: {e}")  
            return pd.Series(name=f"{symbol}_oi", dtype='float64')  
        
        async def fetch_ls_data(symbol: str):  
            try:  
                data = await self.client.futures_global_longshort_ratio(  
                    symbol=symbol, period='4h', limit=500  
                )  
                if data:  
                    df = pd.DataFrame(data)  
                    time_col = 'timestamp' if 'timestamp' in df.columns else 'time'  
                    df[time_col] = pd.to_datetime(df[time_col], unit='ms')  
                    df['lsRatio'] = pd.to_numeric(df['longShortRatio'], errors='coerce')  
                    return df.set_index(time_col)['lsRatio'].rename(f"{symbol}_ls")  
            except Exception as e:  
                print(f"Error fetching L/S ratio for {symbol}: {e}")  
            return pd.Series(name=f"{symbol}_ls", dtype='float64')   
          
        batch_size = 20  
        all_oi_data = []  
        all_ls_data = []  
          
        for i in range(0, len(symbols), batch_size):  
            batch = symbols[i:i + batch_size]  
              
            oi_tasks = [fetch_oi_data(symbol) for symbol in batch]  
            oi_results = await asyncio.gather(*oi_tasks)  
            all_oi_data.extend(oi_results)  
              
            ls_tasks = [fetch_ls_data(symbol) for symbol in batch]  
            ls_results = await asyncio.gather(*ls_tasks)  
            all_ls_data.extend(ls_results)  
              
            await asyncio.sleep(0.5)  
          
        oi_df = pd.concat(all_oi_data, axis=1).sort_index()  
        ls_df = pd.concat(all_ls_data, axis=1).sort_index()  
          
        return {'open_interest': oi_df, 'ls_ratio': ls_df}  


def download_missing_data(
    symbols: List[str], 
    data_dir: str, 
    start_date: str = '2020-01-01', 
    end_date: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Downloads funding rate, open interest, and long-short ratio data if not cached locally,
    and returns them as DataFrames. Incremental updates are performed if some symbols are missing.
    """
    os.makedirs(data_dir, exist_ok=True)
    
    funding_path = os.path.join(data_dir, "futures_data_funding_rate.csv.gz")
    oi_path = os.path.join(data_dir, "futures_data_open_interest.csv.gz")
    ls_path = os.path.join(data_dir, "futures_data_ls_ratio.csv.gz")
    
    # Check if files exist
    funding_exists = os.path.exists(funding_path)
    oi_exists = os.path.exists(oi_path)
    ls_exists = os.path.exists(ls_path)
    
    funding_df = pd.DataFrame()
    oi_df = pd.DataFrame()
    ls_df = pd.DataFrame()
    
    if funding_exists and oi_exists and ls_exists:
        try:
            print("Reading cached futures data from CSV files...")
            funding_df = pd.read_csv(funding_path, compression='gzip', index_col=0, parse_dates=True)
            oi_df = pd.read_csv(oi_path, compression='gzip', index_col=0, parse_dates=True)
            ls_df = pd.read_csv(ls_path, compression='gzip', index_col=0, parse_dates=True)
            
            # Find symbols that are missing from cached data
            missing_funding = [s for s in symbols if s not in funding_df.columns]
            missing_oi = [s for s in symbols if f"{s}_oi" not in oi_df.columns]
            missing_ls = [s for s in symbols if f"{s}_ls" not in ls_df.columns]
            
            missing_symbols = list(set(missing_funding + missing_oi + missing_ls))
            
            if not missing_symbols:
                print("All requested symbols are already cached.")
                return {
                    'funding': funding_df[symbols],
                    'open_interest': oi_df[[f"{s}_oi" for s in symbols if f"{s}_oi" in oi_df.columns]],
                    'ls_ratio': ls_df[[f"{s}_ls" for s in symbols if f"{s}_ls" in ls_df.columns]]
                }
            else:
                print(f"Missing cached data for {len(missing_symbols)} symbols: {missing_symbols}. Downloading...")
                symbols_to_download = missing_symbols
        except Exception as e:
            print(f"Error reading cached files: {e}. Re-downloading all...")
            symbols_to_download = symbols
    else:
        symbols_to_download = symbols
            
    # Apply nest_asyncio to support running event loop in notebook / script environments
    try:
        loop = asyncio.get_running_loop()
        nest_asyncio.apply()
    except RuntimeError:
        pass
        
    async def run_download(syms: List[str]):
        start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
        end_ts = int(pd.to_datetime(end_date or datetime.now()).timestamp() * 1000)
        
        print(f"Downloading futures data from Binance for {len(syms)} symbols...")
        async with FuturesDataCollector() as collector:
            funding_new = await collector.get_funding_rate_batch(syms, start_ts, end_ts)
            risk_new = await collector.get_oi_ls_ratio_batch(syms)
            return funding_new, risk_new['open_interest'], risk_new['ls_ratio']
            
    try:
        funding_new, oi_new, ls_new = asyncio.run(run_download(symbols_to_download))
        
        # Merge with cached data
        if not funding_df.empty:
            funding_df = pd.concat([funding_df, funding_new], axis=1).sort_index()
            # Remove duplicated columns if any
            funding_df = funding_df.loc[:, ~funding_df.columns.duplicated()]
        else:
            funding_df = funding_new
            
        if not oi_df.empty:
            oi_df = pd.concat([oi_df, oi_new], axis=1).sort_index()
            oi_df = oi_df.loc[:, ~oi_df.columns.duplicated()]
        else:
            oi_df = oi_new
            
        if not ls_df.empty:
            ls_df = pd.concat([ls_df, ls_new], axis=1).sort_index()
            ls_df = ls_df.loc[:, ~ls_df.columns.duplicated()]
        else:
            ls_df = ls_new
        
        # Save to local CSV
        funding_df.to_csv(funding_path, compression='gzip')
        oi_df.to_csv(oi_path, compression='gzip')
        ls_df.to_csv(ls_path, compression='gzip')
        print(f"Updated cached data saved to {data_dir}")
        
    except Exception as e:
        print(f"Failed to download data from Binance: {e}.")
        # Fallback for missing symbols by creating empty columns if not present
        for s in symbols:
            if s not in funding_df.columns:
                funding_df[s] = np.nan
            if f"{s}_oi" not in oi_df.columns:
                oi_df[f"{s}_oi"] = np.nan
            if f"{s}_ls" not in ls_df.columns:
                ls_df[f"{s}_ls"] = np.nan
                
    # Return only the requested symbols
    res_funding = funding_df[[s for s in symbols if s in funding_df.columns]]
    res_oi = oi_df[[f"{s}_oi" for s in symbols if f"{s}_oi" in oi_df.columns]]
    res_ls = ls_df[[f"{s}_ls" for s in symbols if f"{s}_ls" in ls_df.columns]]
    
    return {
        'funding': res_funding,
        'open_interest': res_oi,
        'ls_ratio': res_ls
    }

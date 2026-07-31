import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from numba import njit

class CrossSectionalFactorEngine:
    """
    Class để tính toán, tổ hợp và chuyển đổi trọng số cho các yếu tố định lượng 
    crypto (Momentum, Retail Flow, Carry, Margin Risk) theo phương pháp Cross-Sectional.
    """
    
    def __init__(self, symbols: List[str], quantiles: int = 20):
        self.symbols = symbols
        self.quantiles = quantiles
        self.factors_data: Dict[str, pd.DataFrame] = {} # Lưu trữ dữ liệu yếu tố thô (raw factors)
        self.aligned_data: Dict[str, pd.DataFrame] = {}

    def _unpivot_wide_df(self, wide_df: pd.DataFrame, value_name: str, symbol_suffix: str) -> pd.DataFrame:
        """Chuyển đổi DataFrame từ cột rộng (wide format) sang MultiIndex (long format)."""
        if wide_df.empty:
            return pd.DataFrame(columns=[value_name], index=pd.MultiIndex.from_tuples([], names=['Time', 'Symbol']))
            
        wide_df = wide_df.copy()
        
        if isinstance(wide_df.index, pd.DatetimeIndex):
            wide_df.index.name = 'Time'
        else:
            time_col = wide_df.columns[0]
            if time_col in wide_df.columns:
                wide_df = wide_df.set_index(time_col)
            wide_df.index.name = 'Time'
            
        wide_df.index = pd.to_datetime(wide_df.index)
        
        # Xóa hậu tố ('_ls', '_oi') và Unpivot
        new_columns = {col: col.replace(symbol_suffix, '') for col in wide_df.columns if symbol_suffix in col or col in self.symbols}
        df_cleaned = wide_df.rename(columns=new_columns)
        
        # Filter columns to only keep those in self.symbols
        if self.symbols:
            cols_to_keep = [col for col in df_cleaned.columns if col in self.symbols]
            df_cleaned = df_cleaned[cols_to_keep]
            
        # Unpivot (chuyển cột thành hàng)
        df_long = df_cleaned.stack().to_frame(value_name)
        
        # Đổi tên Index và sắp xếp lại thành MultiIndex chuẩn (Symbol, Time)
        df_long.index.names = ['Time', 'Symbol']
        df_long = df_long.swaplevel(0, 1).sort_index()
        
        return df_long

    def _preprocess_data(
        self, 
        klines_dict: Dict[str, pd.DataFrame], 
        ls_ratio_df: pd.DataFrame, 
        oi_df: pd.DataFrame, 
        funding_df: pd.DataFrame
    ) -> None:
        """
        Chuẩn hóa 4 nguồn dữ liệu đầu vào thành MultiIndex DF thống nhất.
        Nếu self.symbols rỗng, nó sẽ được khởi tạo từ Kline data.
        """
        print("Bắt đầu chuẩn hóa dữ liệu đầu vào...")
        
        # 1. Kline Data (Dict -> MultiIndex)
        kline_list = []
        
        if not self.symbols:
             self.symbols = list(klines_dict.keys())
             print(f"  -> Tự động xác định {len(self.symbols)} symbols từ Kline data.")
             
        for symbol, df in klines_dict.items():
            if symbol in self.symbols:
                df = df.copy()
                df['Symbol'] = symbol
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                
                # Make sure the index has a name or defaults to date/time
                idx_name = df.index.name if df.index.name else 'time'
                df.index.name = idx_name
                kline_list.append(df.set_index('Symbol', append=True).reorder_levels(['Symbol', idx_name]))

        self.aligned_data['klines'] = pd.concat(kline_list).sort_index() if kline_list else pd.DataFrame()
        print(f"  -> Kline DF chuẩn hóa: {self.aligned_data['klines'].shape}")

        # 2. Funding Rate (Wide -> MultiIndex)
        df_funding_long = self._unpivot_wide_df(funding_df, value_name='fundingRate', symbol_suffix='')
        self.aligned_data['funding'] = df_funding_long
        print(f"  -> Funding DF chuẩn hóa: {self.aligned_data['funding'].shape}")
        
        # 3 & 4. Long/Short Ratio & Open Interest (Wide -> MultiIndex)
        df_ls_long = self._unpivot_wide_df(ls_ratio_df, value_name='lsRatio', symbol_suffix='_ls')
        df_oi_long = self._unpivot_wide_df(oi_df, value_name='openInterest', symbol_suffix='_oi')
        
        # 5. Hợp nhất OI và LS Ratio (Tạo Margin Risk DF)
        risk_df_temp = df_oi_long.merge(df_ls_long, left_index=True, right_index=True, how='outer')
        self.aligned_data['risk'] = risk_df_temp.sort_index()
        print(f"  -> Risk DF (OI/LS) chuẩn hóa: {self.aligned_data['risk'].shape}") 

    ## --- A. Factor Calculation Methods (Tạo ra yếu tố thô) ---

    def calculate_momentum(self, klines_df: pd.DataFrame, params: Dict) -> pd.Series:
        """Sử dụng mã RSI Bounds của bạn để tính yếu tố Momentum."""
        df = klines_df.copy()
        p = params
        
        # OHLC4
        df['ohlc4'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

        # WMA
        df['ma_base'] = self._custom_wma(df['ohlc4'], p['ma_length'])

        # ATR (Volatility)
        df['volatility'] = self._calculate_atr(df, p['ma_length'])

        # Bounds
        df['upper_bound'], df['lower_bound'] = self._calculate_bounds(
            df['ma_base'], df['volatility'], p['rsi_lower'], p['rsi_upper']
        )

        # Numba state -> Đây là tín hiệu vị thế (1, -1, 0)
        pos_arr = self._trailing_state_numba(
            df['ohlc4'].values,
            df['upper_bound'].values,
            df['lower_bound'].values,
            df['close'].values
        )
        return pd.Series(pos_arr, index=df.index).rename('EnhancedMomentum')

    def calculate_carry(self, funding_series: pd.Series, volatility_series: pd.Series, symbol: str, window: int = 60) -> pd.Series:
        """Tính toán yếu tố Carry Proxy cho MỘT Symbol. FALLBACK nếu data thiếu."""
        if funding_series.empty or len(funding_series.dropna()) < window:
            return pd.Series(index=volatility_series.index, name='EnhancedCarry', dtype=float)
            
        funding_series = funding_series.dropna()
        annualized_funding = funding_series * 3 * 365
        funding_ma = annualized_funding.rolling(window=window).mean()
        funding_anomaly = annualized_funding - funding_ma
        
        vol_aligned = volatility_series.reindex(funding_series.index, method='ffill')
        carry_factor = (annualized_funding / vol_aligned.replace(0, np.nan)) + funding_anomaly
        return carry_factor.rename('EnhancedCarry')

    def calculate_margin_risk(self, risk_df_symbol: pd.DataFrame, volatility_series: pd.Series, symbol: str, window: int = 30) -> pd.Series:
        """Tính toán yếu tố Margin Risk Proxy cho MỘT Symbol. FALLBACK nếu data thiếu."""
        if risk_df_symbol.empty or 'openInterest' not in risk_df_symbol.columns or len(risk_df_symbol.dropna(how='all')) < window:
            return pd.Series(index=volatility_series.index, name='MarginRisk', dtype=float)
        
        risk_df_symbol = risk_df_symbol.dropna(subset=['openInterest', 'lsRatio'])
        if len(risk_df_symbol) < window:
            return pd.Series(index=volatility_series.index, name='MarginRisk', dtype=float)

        # 2. Open Interest Z-Score
        oi = risk_df_symbol['openInterest']
        oi_zscore = (oi - oi.rolling(window=window).mean()) / oi.rolling(window=window).std().replace(0, np.nan)
        
        # 3. Long/Short Imbalance
        ls_imbalance = abs(risk_df_symbol['lsRatio'] - 1.0) 
        
        vol_aligned = volatility_series.reindex(risk_df_symbol.index, method='ffill')
        margin_risk_factor = (oi_zscore * ls_imbalance) / vol_aligned.replace(0, np.nan)
        return margin_risk_factor.rename('MarginRisk')
    
    ## --- B. Factor Processing Methods (Xử lý chéo thị trường) ---

    def _get_quantile_bins(self, s: pd.Series, quantiles: int, long_weight: float = 1.0) -> pd.Series:
        """
        Tính toán các ngưỡng Quantile (chỉ cho nội bộ).
        """
        if s.dropna().empty or len(s.dropna()) < quantiles:
            return pd.Series(0.0, index=s.index)

        quantile_size = 1 / quantiles
        lower_threshold = s.quantile(quantile_size, interpolation='lower')
        upper_threshold = s.quantile(1 - quantile_size, interpolation='higher')

        weights = pd.Series(0.0, index=s.index)
        weights[s >= upper_threshold] = long_weight
        weights[s <= lower_threshold] = -long_weight
        return weights

    def create_cross_sectional_bins(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """
        Thực hiện Binning chéo thị trường, chuyển Factors thành Trọng số Market Neutral Float.
        Mỗi vế (Long/Short) được chuẩn hóa tổng trọng số bằng +1.0 và -1.0.
        """
        factor_data = factor_data.replace([np.inf, -np.inf], np.nan)
        
        # Tie-Breaking Percentile Ranking
        rank_pct = factor_data.rank(axis=1, pct=True, method='first')
        
        # Limit top_pct between 5% and 25% for larger universes, fallback to 50% for small test universes
        num_assets = len(factor_data.columns)
        if num_assets <= 4:
            top_pct = 0.5
        else:
            top_pct = min(0.25, max(0.05, 1.0 / float(self.quantiles)))
        long_mask = rank_pct > (1.0 - top_pct)
        short_mask = rank_pct <= top_pct
        
        final_weights = pd.DataFrame(0.0, index=factor_data.index, columns=factor_data.columns)
        final_weights[long_mask] = 1.0
        final_weights[short_mask] = -1.0
        
        # Normalize each row so Long leg sums to +1.0 and Short leg sums to -1.0
        long_counts = (final_weights > 0).sum(axis=1).replace(0, 1)
        short_counts = (final_weights < 0).sum(axis=1).replace(0, 1)
        
        long_part = final_weights.clip(lower=0.0).div(long_counts, axis=0)
        short_part = final_weights.clip(upper=0.0).div(short_counts, axis=0)
        
        final_weights = (long_part + short_part).fillna(0.0)
        return final_weights

    def ensemble_and_final_bin(self) -> pd.DataFrame:
        """Thực hiện Ensemble Factor và Final Binning (Theo quy trình Quant chuẩn)."""
        if not self.factors_data:
            print("Lỗi: Không có dữ liệu yếu tố thô hợp lệ nào được tính toán. Trả về DataFrame rỗng.")
            return pd.DataFrame()
        
        # 1. Hợp nhất dữ liệu yếu tố thô thành DF MultiIndex (Symbol, Time)
        all_factors_df = pd.concat(self.factors_data.values()).sort_index()

        # 2. Unstack thành Index: Time, Columns: MultiIndex (Factor Name, Symbol)
        transposed_factors = all_factors_df.unstack(level='Symbol')
        transposed_factors.columns.names = ['Factor', 'Symbol']
        
        factors_list = transposed_factors.columns.get_level_values('Factor').unique().tolist()
        binned_factors = []
        
        print("\n--- 3.1 Binning Từng Factor Thô (Cross-Sectional) ---")
        
        for factor_name in factors_list:
            factor_df = transposed_factors[factor_name] 
            binned_factor_df = self.create_cross_sectional_bins(factor_df)
            binned_factor_df = binned_factor_df.rename(columns={col: f"{col}_{factor_name}" for col in binned_factor_df.columns})
            binned_factors.append(binned_factor_df)
            
        is_single_factor = len(factors_list) == 1

        if is_single_factor:
            factor_ensemble = binned_factors[0]
        else:
            print("  -> 3.2 Ensemble: Trung bình cộng các Factors đã Binning.")
            binned_factors_combined = pd.concat(binned_factors, axis=1)
            symbols_only_columns = [col.split('_')[0] for col in binned_factors_combined.columns]
            binned_factors_combined.columns = symbols_only_columns
            factor_ensemble = binned_factors_combined.groupby(by=binned_factors_combined.columns, axis=1).mean()
        
        # 5. Final Binning (Portfolio Neutralization)
        if is_single_factor:
            binned_portfolio = factor_ensemble
            print("  -> 3.3 Trọng số cuối cùng = Factor đã Binning.")
        else:
            print("  -> 3.3 Final Binning: Áp dụng Cross-Sectional Binning lên Ensemble Factor.")
            binned_portfolio = self.create_cross_sectional_bins(factor_ensemble)
            
        return binned_portfolio.rename_axis(columns=None).rename_axis(index=None)

    def calculate_market_cap_proxy(self, klines_all: pd.DataFrame, top_n: int = 40) -> pd.Series:
        """
        Tính toán Market Cap Proxy (GTGD) và lọc ra Top N symbols.
        """
        if klines_all.empty:
            return pd.Series([], dtype=float)
            
        klines_temp = klines_all.copy()
        klines_temp['GTGD'] = klines_temp['volume'] * klines_temp['close'] 
        gtgd_series = klines_temp['GTGD'] # MultiIndex (Symbol, Time)
        
        # Unstack Symbol để tính rolling trên các cột
        gtgd_unstacked = gtgd_series.unstack(level='Symbol') 
        gtgd_ma_unstacked = gtgd_unstacked.rolling(window=30, min_periods=10).mean()
        
        # Stack lại để chuyển về MultiIndex (Time, Symbol) -> Swaplevel -> (Symbol, Time)
        gtgd_ma = gtgd_ma_unstacked.stack(future_stack=True)
        gtgd_ma = gtgd_ma.swaplevel(0, 1).sort_index()

        gtgd_mean_by_symbol = gtgd_series.groupby(level='Symbol').mean()
        top_symbols = gtgd_mean_by_symbol.nlargest(top_n).index.tolist()
        
        if self.symbols != top_symbols:
            print(f"  -> Vũ trụ symbols được giới hạn lại còn {len(top_symbols)} (Top {top_n} GTGD).")
            self.symbols = top_symbols
        
        gtgd_top_n = gtgd_series.loc[top_symbols]
        gtgd_ma_top_n = gtgd_ma.loc[top_symbols]

        market_cap_proxy_factor = (gtgd_top_n / gtgd_ma_top_n).rename('MarketCapProxy')
        return market_cap_proxy_factor

    @staticmethod
    def calculate_retail_flow_proxy(
        klines: pd.DataFrame, 
        volume_period: int = 20, 
        retail_threshold: float = 0.05
    ) -> pd.Series:
        """Tính toán yếu tố Retail Flow Proxy dựa trên Volume và Log Return."""
        log_ret = np.log(klines['close'] / klines['close'].shift(1))
        volatility = log_ret.rolling(window=volume_period).std()
        volume_ma = klines['volume'].rolling(window=volume_period).mean()
        volume_ratio = klines['volume'] / volume_ma
        
        retail_flow_factor = -(log_ret * (volume_ratio / volatility.replace(0, np.nan)))
        return retail_flow_factor.rename('RetailFlow')

    def run_factor_engine(
        self, 
        klines_dict: Dict[str, pd.DataFrame], 
        ls_ratio_df: pd.DataFrame, 
        oi_df: pd.DataFrame, 
        funding_df: pd.DataFrame,
        momentum_params: Dict,
        retail_params: Dict,
        carry_window: int = 60,
        risk_window: int = 30,
        top_n_symbols: Optional[int] = 40
    ) -> pd.DataFrame:
        """Thực hiện toàn bộ quá trình tính toán và tổ hợp yếu tố chéo thị trường."""
        self._preprocess_data(klines_dict, ls_ratio_df, oi_df, funding_df)
        klines_all = self.aligned_data['klines']
        
        if klines_all.empty:
            print("Lỗi: Không có dữ liệu Kline hợp lệ sau khi chuẩn hóa. Dừng tính toán.")
            return pd.DataFrame()

        factor_mcap_proxy = self.calculate_market_cap_proxy(klines_all, top_n=top_n_symbols)
        
        klines_all = klines_all[klines_all.index.get_level_values('Symbol').isin(self.symbols)]
        funding_all = self.aligned_data['funding']
        risk_all = self.aligned_data['risk']
        
        factors_list = []
        print(f"\nBắt đầu tính toán yếu tố thô cho {len(self.symbols)} symbols...")
        
        for symbol in self.symbols:
            try:
                kline_df = klines_all.loc[symbol]
            except KeyError:
                continue

            # Volatility (ATR)
            log_ret = np.log(kline_df['close'] / kline_df['close'].shift(1))
            volatility_daily = log_ret.rolling(window=momentum_params['ma_length']).std().rename('Volatility')
            
            # 1. Momentum
            factor_mom = self.calculate_momentum(kline_df, momentum_params)

            # 2. Retail Flow
            factor_retail = self.calculate_retail_flow_proxy(kline_df, retail_params.get('volume_period', 20))
            
            # 3. Carry
            if symbol in funding_all.index.get_level_values('Symbol'):
                funding_series = funding_all.loc[symbol]['fundingRate']
            else:
                funding_series = pd.Series(dtype=float)
            factor_carry = self.calculate_carry(funding_series, volatility_daily, symbol, carry_window)
            
            # 4. Margin Risk
            if symbol in risk_all.index.get_level_values('Symbol'):
                risk_df_symbol = risk_all.loc[symbol]
            else:
                risk_df_symbol = pd.DataFrame()
            factor_risk = self.calculate_margin_risk(risk_df_symbol, volatility_daily, symbol, risk_window)
            
            # 5. Market Cap Proxy
            try:
                mcap_series = factor_mcap_proxy.loc[symbol]
            except KeyError:
                mcap_series = pd.Series(np.nan, index=kline_df.index, name='MarketCapProxy')
            
            factors_df = pd.concat([factor_mom, factor_retail, factor_carry, factor_risk, mcap_series], axis=1, join='outer')
            factors_df_daily = factors_df.resample('1D').last()
            
            if factors_df_daily.dropna(how='all').empty:
                continue
                
            factors_df_daily['Symbol'] = symbol
            factors_df_daily = factors_df_daily.set_index('Symbol', append=True).reorder_levels(['Symbol', factors_df_daily.index.name])
            self.factors_data[symbol] = factors_df_daily

        return self.ensemble_and_final_bin()
    
    @staticmethod
    @njit(cache=True)
    def _wma_numba(values: np.ndarray, window: int) -> float:
        if len(values) < window:
            return np.nan
        weights = np.arange(1, window + 1).astype(np.float64)
        return np.sum(values[-window:] * weights) / weights.sum()

    @classmethod
    def _custom_wma(cls, series: pd.Series, window: int) -> pd.Series:
        arr = series.values
        result = np.full(len(arr), np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = cls._wma_numba(arr[:i+1], window)
        return pd.Series(result, index=series.index)

    @staticmethod
    def _calculate_atr(data: pd.DataFrame, length: int) -> pd.Series:
        high = data['high']
        low = data['low']
        close = data['close']
        prev_close = close.shift(1)
        tr = np.maximum(high - low, np.maximum(abs(high - prev_close), abs(low - prev_close)))
        atr = tr.ewm(alpha=1/length, adjust=False).mean()
        return atr

    @staticmethod
    def _calculate_bounds(ma: pd.Series, volatility: pd.Series, rsi_lower: int, rsi_upper: int) -> tuple:
        upper = ma + (rsi_upper - 50) / 10 * volatility
        lower = ma - (50 - rsi_lower) / 10 * volatility
        return upper, lower

    @staticmethod
    @njit(cache=True)
    def _trailing_state_numba(
        ohlc4: np.ndarray,
        upper: np.ndarray,
        lower: np.ndarray,
        close: np.ndarray
    ) -> np.ndarray:
        n = len(ohlc4)
        pos = np.zeros(n)
        is_bullish = False
        is_bearish = False

        for i in range(1, n):
            crossover = (ohlc4[i] > upper[i]) and (ohlc4[i-1] <= upper[i-1])
            crossunder = (close[i] < lower[i]) and (close[i-1] >= lower[i-1])

            if crossover and not is_bullish:
                is_bullish = True
                is_bearish = False
                pos[i] = 1
            elif crossunder and not is_bearish:
                is_bullish = is_bearish = False # Reset if crossunder
                is_bearish = True
                pos[i] = -1
            else:
                pos[i] = pos[i-1] if i > 0 else 0

            if pos[i] == 0:
                is_bullish = is_bearish = False

        return pos

    @classmethod
    def calculate_volume_delta_proxy(cls, klines: pd.DataFrame) -> pd.Series:
        """
        Intraday Cumulative Volume Delta (CVD) Proxy.
        CVD_approx = Volume * (2 * Close - High - Low) / (High - Low)
        """
        high = klines['high']
        low = klines['low']
        close = klines['close']
        volume = klines['volume']
        
        denom = (high - low).replace(0, np.nan)
        cvd = volume * (2 * close - high - low) / denom
        return cvd.fillna(0.0)

    @classmethod
    def calculate_open_interest_proxy(cls, klines: pd.DataFrame, window: int = 30) -> pd.Series:
        """
        Open Interest Proxy.
        OI_approx = Rolling Mean(Volume * Close, window) / Rolling Std(Return, window)
        """
        value = klines['volume'] * klines['close']
        oi_mean = value.rolling(window=window, min_periods=min(5, window)).mean()
        
        log_ret = np.log(klines['close'] / klines['close'].shift(1))
        volatility = log_ret.rolling(window=window, min_periods=min(5, window)).std().replace(0, np.nan)
        
        oi_proxy = oi_mean / volatility
        return oi_proxy.ffill().fillna(0.0)

    @classmethod
    def generate_features_for_symbol(
        cls,
        kline_df: pd.DataFrame,
        funding_series: pd.Series,
        windows: List[int] = [7, 14, 30, 60, 90]
    ) -> pd.DataFrame:
        """
        Tạo toàn bộ bộ Features đa khung thời gian cho một Symbol dựa trên OHLCV và Funding Rate.
        """
        kline_df = kline_df.sort_index()
        close = kline_df['close']
        high = kline_df['high']
        low = kline_df['low']
        volume = kline_df['volume']
        
        cvd_proxy = cls.calculate_volume_delta_proxy(kline_df)
        features_dict = {}
        log_ret = np.log(close / close.shift(1))
        
        funding_aligned = funding_series.reindex(kline_df.index).fillna(0.0)
        annualized_funding = funding_aligned * 3 * 365
        
        for w in windows:
            vol = log_ret.rolling(window=w, min_periods=min(3, w)).std()
            vol_clean = vol.replace(0, np.nan)
            
            # --- A. MOMENTUM FEATURE ---
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=w, min_periods=min(3, w)).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=w, min_periods=min(3, w)).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            
            wma = cls._custom_wma(close, w)
            atr = cls._calculate_atr(kline_df, w)
            
            features_dict[f'mom_rsi_{w}'] = rsi.fillna(50)
            features_dict[f'mom_wma_dist_{w}'] = ((close - wma) / (atr.replace(0, np.nan))).fillna(0.0)
            
            # --- B. RETAIL FLOW FEATURE (Contrarian) ---
            volume_ma = volume.rolling(window=w, min_periods=min(3, w)).mean()
            volume_ratio = volume / volume_ma.replace(0, np.nan)
            retail_flow = -(log_ret * (volume_ratio / vol_clean))
            features_dict[f'retail_flow_{w}'] = retail_flow.fillna(0.0)
            
            # --- C. CARRY FEATURE ---
            funding_ma = annualized_funding.rolling(window=w, min_periods=min(3, w)).mean()
            funding_anomaly = annualized_funding - funding_ma
            vol_annual = vol * np.sqrt(365)
            carry = (annualized_funding / vol_annual.replace(0, np.nan)) + funding_anomaly
            features_dict[f'carry_{w}'] = carry.fillna(0.0)
            
            # --- D. MARGIN RISK FEATURE (Proxy-based) ---
            oi_proxy = cls.calculate_open_interest_proxy(kline_df, window=w)
            oi_ma = oi_proxy.rolling(window=w, min_periods=min(3, w)).mean()
            oi_std = oi_proxy.rolling(window=w, min_periods=min(3, w)).std().replace(0, np.nan)
            oi_zscore = (oi_proxy - oi_ma) / oi_std
            
            cvd_ma = cvd_proxy.rolling(window=w, min_periods=min(3, w)).mean()
            vol_sum = volume.rolling(window=w, min_periods=min(3, w)).mean().replace(0, np.nan)
            cvd_imbalance = abs(cvd_ma / vol_sum)
            
            margin_risk = (oi_zscore * cvd_imbalance) / vol_clean
            features_dict[f'margin_risk_{w}'] = margin_risk.fillna(0.0)
            
            # --- E. ENRICHED V3 FEATURES (STRICTLY PAST DATA ONLY) ---
            # 1. Momentum Quality Ratio (Return_w / Volatility_w)
            ret_w = (close / close.shift(w) - 1.0)
            features_dict[f'mom_quality_{w}'] = (ret_w / vol_clean).fillna(0.0)

            # 2. Volume Flow Imbalance Ratio (Volume_w / Volume_30d)
            vol_30d = volume.rolling(window=30, min_periods=5).mean().replace(0, np.nan)
            features_dict[f'vol_imbalance_{w}'] = (volume_ma / vol_30d).fillna(1.0)

            # 3. Funding Rate Divergence (Funding_t - SMA(Funding, w))
            features_dict[f'funding_div_{w}'] = (annualized_funding - funding_ma).fillna(0.0)
            
        features_df = pd.DataFrame(features_dict, index=kline_df.index)
        return features_df

    @classmethod
    def generate_macro_features(cls, macro_df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
        """
        Tạo các Features vĩ mô đa khung thời gian từ DataFrame thô.
        """
        macro_features = {}
        for col in macro_df.columns:
            series = macro_df[col]
            for w in windows:
                macro_features[f'macro_{col}_mean_{w}'] = series.rolling(window=w, min_periods=min(3, w)).mean()
                macro_features[f'macro_{col}_std_{w}'] = series.rolling(window=w, min_periods=min(3, w)).std()
                if col in ['stablecoin_mcap', 'spy', 'dxy']:
                    macro_features[f'macro_{col}_roc_{w}'] = (series / series.shift(w) - 1.0)
        return pd.DataFrame(macro_features, index=macro_df.index).fillna(0.0)

    @classmethod
    def prepare_panel_dataset(
        cls,
        data_dict: Dict[str, pd.DataFrame],
        funding_df: pd.DataFrame,
        symbols: List[str],
        macro_df: pd.DataFrame,
        windows: List[int] = [7, 14, 30, 60, 90],
        lag: int = 1
    ) -> pd.DataFrame:
        """
        Tạo Panel Dataset chứa features của tất cả active symbols và target forward return (kèm features vĩ mô).
        Thực hiện Target Demeanization và Cross-Sectional Percentile Ranking trên các đặc trưng riêng của coin.
        """
        macro_features_df = pd.DataFrame()
        if not macro_df.empty:
            macro_features_df = cls.generate_macro_features(macro_df, windows)
            
        panel_list = []
        funding_daily = pd.DataFrame()
        if not funding_df.empty:
            funding_daily = funding_df.resample('1D').last()
            
        # Xác định các cột đặc trưng riêng của coin
        sample_symbol = symbols[0] if symbols else None
        asset_feature_names = []
        
        for symbol in symbols:
            if symbol not in data_dict:
                continue
            kline_df = data_dict[symbol].copy()
            if kline_df.empty or len(kline_df) < max(windows):
                continue
                
            if not funding_daily.empty and symbol in funding_daily.columns:
                funding_series = funding_daily[symbol]
            else:
                funding_series = pd.Series(0.0, index=kline_df.index)
                
            features_df = cls.generate_features_for_symbol(kline_df, funding_series, windows)
            if not asset_feature_names:
                asset_feature_names = list(features_df.columns)
            
            # Join macro features
            if not macro_features_df.empty:
                features_df = features_df.join(macro_features_df, how='left')
            
            close = kline_df['close']
            forward_return = (close.shift(-lag) / close - 1).rename('target')
            
            symbol_df = pd.concat([features_df, forward_return], axis=1)
            symbol_df['Symbol'] = symbol
            symbol_df = symbol_df.dropna(subset=['target'])
            
            panel_list.append(symbol_df)
            
        if not panel_list:
            return pd.DataFrame()
            
        panel_df = pd.concat(panel_list)
        panel_df.index.name = 'Time'
        panel_df = panel_df.reset_index().set_index(['Time', 'Symbol']).sort_index()
        
        # 1. Cross-Sectional Z-Score Standardization cho các đặc trưng riêng của coin (Strictly per date timestamp)
        if asset_feature_names:
            def zscore_transform(df_group):
                std = df_group.std()
                if isinstance(std, pd.Series):
                    std = std.replace(0, 1.0).fillna(1.0)
                else:
                    std = 1.0 if (pd.isna(std) or std == 0) else std
                return (df_group - df_group.mean()) / std

            panel_df[asset_feature_names] = (
                panel_df[asset_feature_names]
                .groupby(level='Time')
                .transform(zscore_transform)
                .fillna(0.0)
            )
            
        # 2. Target Demeanization: Neutralize target return against daily market average at same timestamp
        target_mean = panel_df['target'].groupby(level='Time').transform('mean')
        panel_df['target'] = panel_df['target'] - target_mean
        
        return panel_df

    @staticmethod
    def filter_features_by_ic_and_collinearity(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        max_corr_threshold: float = 0.80
    ) -> List[str]:
        """
        Feature Selection Filter:
        1. Computes Spearman Rank IC per feature on training data.
        2. Drops highly collinear features (r > 0.80), keeping the feature with higher Rank IC.
        """
        from scipy.stats import spearmanr
        feature_ic = {}
        for col in X_train.columns:
            r, _ = spearmanr(X_train[col], y_train)
            feature_ic[col] = abs(r) if not np.isnan(r) else 0.0

        sorted_features = sorted(feature_ic.keys(), key=lambda f: feature_ic[f], reverse=True)
        
        selected_features = []
        corr_matrix = X_train.corr().abs()

        for feat in sorted_features:
            keep = True
            for prev_feat in selected_features:
                if corr_matrix.loc[feat, prev_feat] > max_corr_threshold:
                    keep = False
                    break
            if keep:
                selected_features.append(feat)

        return selected_features

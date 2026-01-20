"""
Адаптер, объединяющий бычий и медвежий сканеры для совместимости со старым кодом
"""
from .bullish_flag_scanner import BullishFlagScanner
from .bearish_flag_scanner import BearishFlagScanner
from t_tech.invest import CandleInterval


class ComplexFlagScanner:
    """
    Объединенный сканер, использующий бычий и медвежий сканеры
    Для совместимости со старым кодом
    """
    
    def __init__(self, token: str):
        self.token = token
        self.bullish_scanner = BullishFlagScanner(token)
        self.bearish_scanner = BearishFlagScanner(token)
    
    def get_all_shares(self):
        """Получает список всех доступных акций (использует бычий сканер)"""
        return self.bullish_scanner.get_all_shares()
    
    def get_candles_df(self, ticker: str, class_code: str, days_back: int = 5, interval: CandleInterval = CandleInterval.CANDLE_INTERVAL_HOUR):
        """Загружает свечи (использует бычий сканер)"""
        return self.bullish_scanner.get_candles_df(ticker, class_code, days_back, interval)
    
    def get_candles_by_uid(self, uid: str, days_back: int = 5, interval: CandleInterval = CandleInterval.CANDLE_INTERVAL_HOUR):
        """Загружает свечи по UID (использует бычий сканер)"""
        return self.bullish_scanner.get_candles_by_uid(uid, days_back, interval)
    
    def get_candles_df_by_dates(self, ticker: str, class_code: str, from_date, to_date, interval: CandleInterval = CandleInterval.CANDLE_INTERVAL_HOUR):
        """Загружает свечи по датам (использует бычий сканер)"""
        return self.bullish_scanner.get_candles_df_by_dates(ticker, class_code, from_date, to_date, interval)
    
    def analyze_flag_0_1_2_3_4(self, df, debug=False, timeframe='1h', window=3, scan_type='all', min_pole_pct=None):
        """Анализирует бычий флаг"""
        return self.bullish_scanner.analyze_flag_0_1_2_3_4(df, debug=debug, timeframe=timeframe, window=window, scan_type=scan_type, min_pole_pct=min_pole_pct)
    
    def analyze_bearish_flag_0_1_2_3_4(self, df, debug=False, timeframe='1h', window=3, scan_type='all', min_pole_pct=None):
        """Анализирует медвежий флаг"""
        return self.bearish_scanner.analyze_bearish_flag_0_1_2_3_4(df, debug=debug, timeframe=timeframe, window=window, scan_type=scan_type, min_pole_pct=min_pole_pct)
    
    def analyze(self, df, debug=False, timeframe='1h', window=3, scan_type='all', min_pole_pct=None):
        """Анализирует оба типа паттернов"""
        patterns = []
        bullish = self.bullish_scanner.analyze(df, debug=debug, timeframe=timeframe, window=window, scan_type=scan_type, min_pole_pct=min_pole_pct)
        if bullish:
            patterns.extend(bullish)
        bearish = self.bearish_scanner.analyze(df, debug=debug, timeframe=timeframe, window=window, scan_type=scan_type, min_pole_pct=min_pole_pct)
        if bearish:
            patterns.extend(bearish)
        return patterns

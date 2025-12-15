"""
Модуль оптимизации торговой стратегии на исторических данных
"""
import os
import logging
from decimal import Decimal
from datetime import timedelta
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from t_tech.invest import (
    Client,
    InstrumentIdType,
    CandleInterval,
)
from t_tech.invest.utils import (
    quotation_to_decimal,
    now,
)

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class StrategyResult:
    """Результат тестирования стратегии"""
    strategy_type: str  # 'EMA', 'RSI', 'MACD', 'BB', 'STOCH', 'PATTERN', 'COMBINED'
    pattern_type: str = ""  # Тип паттерна: 'candlestick', 'chart', 'combined'
    pattern_name: str = ""  # Название паттерна: 'hammer', 'engulfing', 'double_bottom', etc.
    ema_short: int = 0
    ema_long: int = 0
    rsi_period: int = 0
    rsi_oversold: float = 0.0
    rsi_overbought: float = 0.0
    macd_fast: int = 0
    macd_slow: int = 0
    macd_signal: int = 0
    bb_period: int = 0
    bb_std: float = 0.0
    stoch_k: int = 0
    stoch_d: int = 0
    timeframe: CandleInterval = CandleInterval.CANDLE_INTERVAL_1_MIN
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    profit_percent: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0


class StrategyOptimizer:
    """Класс для оптимизации торговой стратегии на исторических данных"""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("TINKOFF_INVEST_TOKEN")
        if not self.token:
            raise ValueError("TINKOFF_INVEST_TOKEN не установлен")
        self.client: Optional[Client] = None
        self.instrument = None
        
    def _setup_instrument(self, ticker: str = "SBER", class_code: str = "TQBR"):
        """Настройка инструмента"""
        item = self.client.instruments.get_instrument_by(
            id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_TICKER,
            id=ticker,
            class_code=class_code,
        ).instrument
        self.instrument = item
        logger.info(f"Инструмент: {item.name} (FIGI: {item.figi})")
        return item
    
    def load_candle_data(
        self, 
        days: int = 30, 
        ticker: str = "SBER", 
        class_code: str = "TQBR",
        interval: CandleInterval = CandleInterval.CANDLE_INTERVAL_1_MIN
    ) -> pd.DataFrame:
        """Загрузка данных свечей за указанное количество дней"""
        interval_name = interval.name if hasattr(interval, 'name') else str(interval)
        logger.info(f"Загрузка данных за {days} дней с таймфреймом {interval_name}...")
        
        try:
            with Client(self.token) as client:
                self.client = client
                self._setup_instrument(ticker, class_code)
                
                # Разбиваем запрос на чанки по 7 дней, чтобы избежать таймаутов и обновлять UI
                chunk_days = 7
                total_chunks = (days + chunk_days - 1) // chunk_days
                
                all_candles = []
                end_date = now()
                start_date = end_date - timedelta(days=days)
                
                logger.info(f"Запрос данных разбит на {total_chunks} частей по {chunk_days} дней")
                
                current_start = start_date
                for i in range(total_chunks):
                    current_end = min(current_start + timedelta(days=chunk_days), end_date)
                    
                    logger.info(f"Загрузка части {i+1}/{total_chunks}: {current_start.date()} - {current_end.date()}")
                    
                    try:
                        # Используем get_all_candles для чанка
                        chunk_candles = self.client.get_all_candles(
                            instrument_id=self.instrument.uid,
                            from_=current_start,
                            to=current_end,
                            interval=interval,
                        )
                        
                        # Сразу конвертируем в словари для экономии памяти и объединения
                        for c in chunk_candles:
                            try:
                                all_candles.append({
                                    'time': c.time,
                                    'open': float(quotation_to_decimal(c.open)),
                                    'high': float(quotation_to_decimal(c.high)),
                                    'low': float(quotation_to_decimal(c.low)),
                                    'close': float(quotation_to_decimal(c.close)),
                                    'volume': c.volume
                                })
                            except Exception as e:
                                continue
                                
                    except Exception as e:
                        logger.warning(f"Ошибка при загрузке чанка {i+1}: {e}")
                        # Не прерываем, пробуем следующий чанк
                    
                    current_start = current_end
                
                if not all_candles:
                    raise ValueError(f"Не удалось загрузить данные для {ticker}. Проверьте тикер и доступность данных.")
                
                df = pd.DataFrame(all_candles)
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
                df.sort_index(inplace=True)
                
                # Удаляем дубликаты по времени
                df = df[~df.index.duplicated(keep='first')]
                
                logger.info(f"Всего загружено {len(df)} свечей")
                
                if len(df) < 100:
                    raise ValueError(f"Недостаточно данных: загружено только {len(df)} свечей. Увеличьте период или проверьте доступность данных.")
                
                return df
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            raise
    
    def load_minute_data(self, days: int = 30, ticker: str = "SBER", class_code: str = "TQBR") -> pd.DataFrame:
        """Загрузка минутных данных (для обратной совместимости)"""
        return self.load_candle_data(days, ticker, class_code, CandleInterval.CANDLE_INTERVAL_1_MIN)
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Расчет RSI (Relative Strength Index)"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Расчет MACD (Moving Average Convergence Divergence)"""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """Расчет Bollinger Bands"""
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return pd.DataFrame({
            'bb_upper': upper_band,
            'bb_middle': sma,
            'bb_lower': lower_band
        })
    
    def calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Расчет Stochastic Oscillator"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()
        return pd.DataFrame({
            'stoch_k': k_percent,
            'stoch_d': d_percent
        })
    
    def detect_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Обнаружение свечных паттернов"""
        df = df.copy()
        
        # Вычисляем размеры тел и теней
        df['body'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['range'] = df['high'] - df['low']
        
        # Избегаем деления на ноль
        df['body_ratio'] = df['body'] / (df['range'] + 1e-10)
        df['upper_shadow_ratio'] = df['upper_shadow'] / (df['range'] + 1e-10)
        df['lower_shadow_ratio'] = df['lower_shadow'] / (df['range'] + 1e-10)
        
        # Определяем тип свечи (бычья/медвежья)
        df['is_bullish'] = df['close'] > df['open']
        df['is_bearish'] = df['close'] < df['open']
        
        # Инициализируем колонки для паттернов
        df['pattern'] = ''
        df['pattern_signal'] = 0  # 1 = BUY, -1 = SELL, 0 = HOLD
        
        # Молот (Hammer) - бычий паттерн
        hammer = (
            (df['lower_shadow_ratio'] > 2.0) &  # Длинная нижняя тень
            (df['upper_shadow_ratio'] < 0.3) &  # Короткая верхняя тень
            (df['body_ratio'] < 0.3) &  # Маленькое тело
            (df['range'] > 0)  # Есть движение
        )
        df.loc[hammer, 'pattern'] = 'hammer'
        df.loc[hammer, 'pattern_signal'] = 1
        
        # Перевернутый молот (Inverted Hammer) - бычий паттерн
        inverted_hammer = (
            (df['upper_shadow_ratio'] > 2.0) &  # Длинная верхняя тень
            (df['lower_shadow_ratio'] < 0.3) &  # Короткая нижняя тень
            (df['body_ratio'] < 0.3) &  # Маленькое тело
            (df['range'] > 0)
        )
        df.loc[inverted_hammer, 'pattern'] = 'inverted_hammer'
        df.loc[inverted_hammer, 'pattern_signal'] = 1
        
        # Поглощение (Engulfing)
        prev_bullish = df['is_bullish'].shift(1)
        prev_bearish = df['is_bearish'].shift(1)
        prev_body = df['body'].shift(1)
        prev_open = df['open'].shift(1)
        prev_close = df['close'].shift(1)
        
        # Бычье поглощение
        bullish_engulfing = (
            prev_bearish &  # Предыдущая свеча медвежья
            df['is_bullish'] &  # Текущая свеча бычья
            (df['open'] < prev_close) &  # Открытие ниже закрытия предыдущей
            (df['close'] > prev_open) &  # Закрытие выше открытия предыдущей
            (df['body'] > prev_body * 1.1)  # Тело больше предыдущего
        )
        df.loc[bullish_engulfing, 'pattern'] = 'bullish_engulfing'
        df.loc[bullish_engulfing, 'pattern_signal'] = 1
        
        # Медвежье поглощение
        bearish_engulfing = (
            prev_bullish &  # Предыдущая свеча бычья
            df['is_bearish'] &  # Текущая свеча медвежья
            (df['open'] > prev_close) &  # Открытие выше закрытия предыдущей
            (df['close'] < prev_open) &  # Закрытие ниже открытия предыдущей
            (df['body'] > prev_body * 1.1)  # Тело больше предыдущего
        )
        df.loc[bearish_engulfing, 'pattern'] = 'bearish_engulfing'
        df.loc[bearish_engulfing, 'pattern_signal'] = -1
        
        # Доджи (Doji) - неопределенность
        doji = (
            (df['body_ratio'] < 0.1) &  # Очень маленькое тело
            (df['range'] > 0)
        )
        df.loc[doji, 'pattern'] = 'doji'
        # Доджи сам по себе не дает сигнала
        
        # Звезда (Shooting Star) - медвежий паттерн
        shooting_star = (
            (df['upper_shadow_ratio'] > 2.0) &  # Длинная верхняя тень
            (df['lower_shadow_ratio'] < 0.3) &  # Короткая нижняя тень
            (df['body_ratio'] < 0.3) &  # Маленькое тело
            df['is_bearish'] &  # Медвежья свеча
            (df['range'] > 0)
        )
        df.loc[shooting_star, 'pattern'] = 'shooting_star'
        df.loc[shooting_star, 'pattern_signal'] = -1
        
        # Падающая звезда (Hanging Man) - медвежий паттерн
        hanging_man = (
            (df['lower_shadow_ratio'] > 2.0) &  # Длинная нижняя тень
            (df['upper_shadow_ratio'] < 0.3) &  # Короткая верхняя тень
            (df['body_ratio'] < 0.3) &  # Маленькое тело
            df['is_bearish'] &  # Медвежья свеча
            (df['range'] > 0)
        )
        df.loc[hanging_man, 'pattern'] = 'hanging_man'
        df.loc[hanging_man, 'pattern_signal'] = -1
        
        return df
    
    def detect_chart_patterns(self, df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """Обнаружение графических паттернов"""
        df = df.copy()
        
        # Инициализируем колонки для паттернов
        if 'chart_pattern' not in df.columns:
            df['chart_pattern'] = ''
        if 'chart_pattern_signal' not in df.columns:
            df['chart_pattern_signal'] = 0
        
        # Вычисляем локальные максимумы и минимумы
        df['local_max'] = df['high'].rolling(window=3, center=True).max() == df['high']
        df['local_min'] = df['low'].rolling(window=3, center=True).min() == df['low']
        
        # Двойное дно (Double Bottom) - бычий паттерн
        # Используем более простой подход для избежания проблем с индексами
        local_mins_all = df[df['local_min']].copy()
        
        if len(local_mins_all) >= 2:
            min_values = local_mins_all['low'].values
            min_indices = local_mins_all.index
            
            # Проверяем, есть ли два минимума на похожем уровне
            for j in range(len(min_values) - 1):
                for k in range(j + 1, len(min_values)):
                    if min(min_values[j], min_values[k]) == 0:
                        continue
                    price_diff = abs(min_values[j] - min_values[k]) / min(min_values[j], min_values[k])
                    
                    # Вычисляем разницу во времени
                    try:
                        if isinstance(min_indices[j], pd.Timestamp) and isinstance(min_indices[k], pd.Timestamp):
                            time_diff_hours = abs((min_indices[k] - min_indices[j]).total_seconds() / 3600)
                        else:
                            # Если индексы не datetime, используем позицию
                            time_diff_hours = abs(k - j)
                    except:
                        time_diff_hours = abs(k - j)
                    
                    # Два минимума на похожем уровне (разница < 2%) и с разницей во времени
                    if price_diff < 0.02 and 4 < time_diff_hours < lookback * 24:  # lookback в часах для минутных данных
                        # Проверяем, что между ними был рост
                        try:
                            between = df.loc[min_indices[j]:min_indices[k]]
                            if len(between) > 0:
                                peak = between['high'].max()
                                if peak > min(min_values[j], min_values[k]) * 1.01:  # Был рост минимум на 1%
                                    df.loc[min_indices[k], 'chart_pattern'] = 'double_bottom'
                                    df.loc[min_indices[k], 'chart_pattern_signal'] = 1
                        except:
                            pass
        
        # Двойная вершина (Double Top) - медвежий паттерн
        local_maxs_all = df[df['local_max']].copy()
        
        if len(local_maxs_all) >= 2:
            max_values = local_maxs_all['high'].values
            max_indices = local_maxs_all.index
            
            for j in range(len(max_values) - 1):
                for k in range(j + 1, len(max_values)):
                    if max(max_values[j], max_values[k]) == 0:
                        continue
                    price_diff = abs(max_values[j] - max_values[k]) / max(max_values[j], max_values[k])
                    
                    try:
                        if isinstance(max_indices[j], pd.Timestamp) and isinstance(max_indices[k], pd.Timestamp):
                            time_diff_hours = abs((max_indices[k] - max_indices[j]).total_seconds() / 3600)
                        else:
                            time_diff_hours = abs(k - j)
                    except:
                        time_diff_hours = abs(k - j)
                    
                    if price_diff < 0.02 and 4 < time_diff_hours < lookback * 24:
                        try:
                            between = df.loc[max_indices[j]:max_indices[k]]
                            if len(between) > 0:
                                trough = between['low'].min()
                                if trough < max(max_values[j], max_values[k]) * 0.99:  # Был спад минимум на 1%
                                    df.loc[max_indices[k], 'chart_pattern'] = 'double_top'
                                    df.loc[max_indices[k], 'chart_pattern_signal'] = -1
                        except:
                            pass
        
        # Тройное дно (Triple Bottom) - бычий паттерн
        if len(local_mins_all) >= 3:
            # Берем последние 3 минимума
            min_values = local_mins_all['low'].values[-3:]
            min_indices = local_mins_all.index[-3:]
            
            # Проверяем, что все три минимума на похожем уровне
            if all(min(min_values[0], min_values[j]) > 0 and abs(min_values[0] - min_values[j]) / min(min_values[0], min_values[j]) < 0.02 for j in [1, 2]):
                df.loc[min_indices[-1], 'chart_pattern'] = 'triple_bottom'
                df.loc[min_indices[-1], 'chart_pattern_signal'] = 1
        
        # Тройная вершина (Triple Top) - медвежий паттерн
        if len(local_maxs_all) >= 3:
            max_values = local_maxs_all['high'].values[-3:]
            max_indices = local_maxs_all.index[-3:]
            
            if all(max(max_values[0], max_values[j]) > 0 and abs(max_values[0] - max_values[j]) / max(max_values[0], max_values[j]) < 0.02 for j in [1, 2]):
                df.loc[max_indices[-1], 'chart_pattern'] = 'triple_top'
                df.loc[max_indices[-1], 'chart_pattern_signal'] = -1
        
        return df
    
    def calculate_pattern_signals(self, df: pd.DataFrame, pattern_type: str = "candlestick", pattern_names: List[str] = None) -> pd.DataFrame:
        """Расчет сигналов на основе паттернов"""
        df = df.copy()
        
        if pattern_names is None:
            if pattern_type == "candlestick":
                pattern_names = ['hammer', 'inverted_hammer', 'bullish_engulfing', 'bearish_engulfing', 'shooting_star', 'hanging_man']
            elif pattern_type == "chart":
                pattern_names = ['double_bottom', 'double_top', 'triple_bottom', 'triple_top']
            else:
                pattern_names = []
        
        # Обнаруживаем паттерны
        if pattern_type == "candlestick" or pattern_type == "combined":
            df = self.detect_candlestick_patterns(df)
        
        if pattern_type == "chart" or pattern_type == "combined":
            df = self.detect_chart_patterns(df)
        
        # Генерируем сигналы на основе выбранных паттернов
        df['signal'] = 0
        df['position'] = 0
        
        if pattern_type == "candlestick" or pattern_type == "combined":
            # Используем свечные паттерны
            for pattern_name in pattern_names:
                if pattern_name in ['hammer', 'inverted_hammer', 'bullish_engulfing']:
                    buy_signals = df['pattern'] == pattern_name
                    df.loc[buy_signals, 'signal'] = 1
                elif pattern_name in ['bearish_engulfing', 'shooting_star', 'hanging_man']:
                    sell_signals = df['pattern'] == pattern_name
                    df.loc[sell_signals, 'signal'] = -1
        
        if pattern_type == "chart" or pattern_type == "combined":
            # Используем графические паттерны
            for pattern_name in pattern_names:
                if pattern_name in ['double_bottom', 'triple_bottom']:
                    buy_signals = df['chart_pattern'] == pattern_name
                    df.loc[buy_signals, 'signal'] = 1
                elif pattern_name in ['double_top', 'triple_top']:
                    sell_signals = df['chart_pattern'] == pattern_name
                    df.loc[sell_signals, 'signal'] = -1
        
        # Отслеживаем позицию
        position = 0
        for idx in df.index:
            if df.loc[idx, 'signal'] == 1:
                position = 1
            elif df.loc[idx, 'signal'] == -1:
                position = 0
            df.loc[idx, 'position'] = position
        
        return df
    
    def calculate_ema_signals(self, df: pd.DataFrame, ema_short: int, ema_long: int) -> pd.DataFrame:
        """Расчет EMA и генерация сигналов"""
        df = df.copy()
        
        # Расчет EMA
        df['ema_short'] = df['close'].ewm(span=ema_short, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=ema_long, adjust=False).mean()
        
        # Генерация сигналов
        # 1 = BUY (золотой крест), -1 = SELL (мертвый крест), 0 = HOLD
        df['signal'] = 0
        df['position'] = 0  # 1 = в позиции, 0 = нет позиции
        
        # Находим пересечения
        prev_short = df['ema_short'].shift(1)
        prev_long = df['ema_long'].shift(1)
        
        # Золотой крест (BUY): короткая пересекает длинную снизу вверх
        golden_cross = (prev_short <= prev_long) & (df['ema_short'] > df['ema_long'])
        df.loc[golden_cross, 'signal'] = 1
        
        # Мертвый крест (SELL): короткая пересекает длинную сверху вниз
        death_cross = (prev_short >= prev_long) & (df['ema_short'] < df['ema_long'])
        df.loc[death_cross, 'signal'] = -1
        
        # Отслеживаем позицию
        position = 0
        for idx in df.index:
            if df.loc[idx, 'signal'] == 1:
                position = 1
            elif df.loc[idx, 'signal'] == -1:
                position = 0
            df.loc[idx, 'position'] = position
        
        return df
    
    def calculate_rsi_signals(self, df: pd.DataFrame, period: int = 14, oversold: float = 30.0, overbought: float = 70.0) -> pd.DataFrame:
        """Расчет сигналов на основе RSI"""
        df = df.copy()
        df['rsi'] = self.calculate_rsi(df, period)
        
        df['signal'] = 0
        df['position'] = 0
        
        # BUY: RSI выходит из зоны перепроданности (снизу вверх через 30)
        # SELL: RSI выходит из зоны перекупленности (сверху вниз через 70)
        prev_rsi = df['rsi'].shift(1)
        
        buy_signal = (prev_rsi <= oversold) & (df['rsi'] > oversold)
        sell_signal = (prev_rsi >= overbought) & (df['rsi'] < overbought)
        
        df.loc[buy_signal, 'signal'] = 1
        df.loc[sell_signal, 'signal'] = -1
        
        # Отслеживаем позицию
        position = 0
        for idx in df.index:
            if df.loc[idx, 'signal'] == 1:
                position = 1
            elif df.loc[idx, 'signal'] == -1:
                position = 0
            df.loc[idx, 'position'] = position
        
        return df
    
    def calculate_macd_signals(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Расчет сигналов на основе MACD"""
        df = df.copy()
        macd_data = self.calculate_macd(df, fast, slow, signal)
        df['macd'] = macd_data['macd']
        df['macd_signal'] = macd_data['signal']
        df['macd_histogram'] = macd_data['histogram']
        
        df['signal'] = 0
        df['position'] = 0
        
        # BUY: MACD пересекает сигнальную линию снизу вверх
        # SELL: MACD пересекает сигнальную линию сверху вниз
        prev_macd = df['macd'].shift(1)
        prev_signal = df['macd_signal'].shift(1)
        
        buy_signal = (prev_macd <= prev_signal) & (df['macd'] > df['macd_signal'])
        sell_signal = (prev_macd >= prev_signal) & (df['macd'] < df['macd_signal'])
        
        df.loc[buy_signal, 'signal'] = 1
        df.loc[sell_signal, 'signal'] = -1
        
        # Отслеживаем позицию
        position = 0
        for idx in df.index:
            if df.loc[idx, 'signal'] == 1:
                position = 1
            elif df.loc[idx, 'signal'] == -1:
                position = 0
            df.loc[idx, 'position'] = position
        
        return df
    
    def calculate_bb_signals(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """Расчет сигналов на основе Bollinger Bands"""
        df = df.copy()
        bb_data = self.calculate_bollinger_bands(df, period, std_dev)
        df['bb_upper'] = bb_data['bb_upper']
        df['bb_middle'] = bb_data['bb_middle']
        df['bb_lower'] = bb_data['bb_lower']
        
        df['signal'] = 0
        df['position'] = 0
        
        # BUY: цена касается нижней полосы и отскакивает
        # SELL: цена касается верхней полосы и отскакивает
        prev_close = df['close'].shift(1)
        prev_lower = df['bb_lower'].shift(1)
        prev_upper = df['bb_upper'].shift(1)
        
        buy_signal = (prev_close <= prev_lower) & (df['close'] > df['bb_lower'])
        sell_signal = (prev_close >= prev_upper) & (df['close'] < df['bb_upper'])
        
        df.loc[buy_signal, 'signal'] = 1
        df.loc[sell_signal, 'signal'] = -1
        
        # Отслеживаем позицию
        position = 0
        for idx in df.index:
            if df.loc[idx, 'signal'] == 1:
                position = 1
            elif df.loc[idx, 'signal'] == -1:
                position = 0
            df.loc[idx, 'position'] = position
        
        return df
    
    def calculate_stoch_signals(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3, oversold: float = 20.0, overbought: float = 80.0) -> pd.DataFrame:
        """Расчет сигналов на основе Stochastic"""
        df = df.copy()
        stoch_data = self.calculate_stochastic(df, k_period, d_period)
        df['stoch_k'] = stoch_data['stoch_k']
        df['stoch_d'] = stoch_data['stoch_d']
        
        df['signal'] = 0
        df['position'] = 0
        
        # BUY: %K пересекает %D снизу вверх в зоне перепроданности
        # SELL: %K пересекает %D сверху вниз в зоне перекупленности
        prev_k = df['stoch_k'].shift(1)
        prev_d = df['stoch_d'].shift(1)
        
        buy_signal = (prev_k <= prev_d) & (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'] < oversold + 10)
        sell_signal = (prev_k >= prev_d) & (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'] > overbought - 10)
        
        df.loc[buy_signal, 'signal'] = 1
        df.loc[sell_signal, 'signal'] = -1
        
        # Отслеживаем позицию
        position = 0
        for idx in df.index:
            if df.loc[idx, 'signal'] == 1:
                position = 1
            elif df.loc[idx, 'signal'] == -1:
                position = 0
            df.loc[idx, 'position'] = position
        
        return df
    
    def backtest_strategy(
        self, 
        df: pd.DataFrame, 
        strategy_type: str = "EMA",
        ema_short: int = 5,
        ema_long: int = 20,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        bb_period: int = 20,
        bb_std: float = 2.0,
        stoch_k: int = 14,
        stoch_d: int = 3,
        pattern_type: str = "candlestick",
        pattern_names: List[str] = None,
        timeframe: CandleInterval = CandleInterval.CANDLE_INTERVAL_1_MIN,
        commission_percent: float = 0.05,  # Комиссия в процентах за одну сделку (0.05% = 0.0005)
        take_profit: float = 0.0, # 0.01 = 1% (0.0 = disabled)
        trailing_stop: float = 0.0 # 0.003 = 0.3% (0.0 = disabled)
    ) -> StrategyResult:
        """Бэктестинг стратегии на исторических данных"""
        # Определяем минимальное количество данных в зависимости от стратегии
        min_period = max(ema_long, rsi_period, macd_slow, bb_period, stoch_k, 20)  # Минимум 20 для паттернов
        if len(df) < min_period:
            pattern_name_str = ",".join(pattern_names) if pattern_names else pattern_type
            return StrategyResult(
                strategy_type=strategy_type,
                ema_short=ema_short,
                ema_long=ema_long,
                rsi_period=rsi_period,
                rsi_oversold=rsi_oversold,
                rsi_overbought=rsi_overbought,
                macd_fast=macd_fast,
                macd_slow=macd_slow,
                macd_signal=macd_signal,
                bb_period=bb_period,
                bb_std=bb_std,
                stoch_k=stoch_k,
                stoch_d=stoch_d,
                pattern_type=pattern_type,
                pattern_name=pattern_name_str,
                timeframe=timeframe,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                total_profit=0.0,
                profit_percent=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                win_rate=0.0
            )
        
        # Выбираем стратегию для расчета сигналов
        if strategy_type == "EMA":
            df = self.calculate_ema_signals(df, ema_short, ema_long)
        elif strategy_type == "RSI":
            df = self.calculate_rsi_signals(df, rsi_period, rsi_oversold, rsi_overbought)
        elif strategy_type == "MACD":
            df = self.calculate_macd_signals(df, macd_fast, macd_slow, macd_signal)
        elif strategy_type == "BB":
            df = self.calculate_bb_signals(df, bb_period, bb_std)
        elif strategy_type == "STOCH":
            df = self.calculate_stoch_signals(df, stoch_k, stoch_d)
        elif strategy_type == "PATTERN":
            df = self.calculate_pattern_signals(df, pattern_type, pattern_names)
        else:
            # По умолчанию используем EMA
            df = self.calculate_ema_signals(df, ema_short, ema_long)
        
        # Симуляция торговли
        trades = []
        entry_price = None
        entry_time = None
        
        # Оптимизация: итерируемся только по строкам с сигналами
        signals_df = df[df['signal'] != 0]
        
        # Преобразуем индексы (временные метки) сигналов в список для итерации
        signal_indices = signals_df.index.tolist()
        
        i = 0
        while i < len(signal_indices):
            idx = signal_indices[i]
            row = signals_df.loc[idx]
            signal = row['signal']
            
            # Покупка
            if signal == 1 and entry_price is None:
                entry_price = float(row['close'])
                entry_time = idx
                
                # Если включен риск-менеджмент, проверяем выход по TP/SL до следующего сигнала
                if take_profit > 0 or trailing_stop > 0:
                    # Определяем конец периода проверки (следующий сигнал или конец данных)
                    next_signal_idx = signal_indices[i + 1] if i + 1 < len(signal_indices) else df.index[-1]
                    
                    # Берем срез данных от входа до следующего сигнала
                    # Используем slice по loc, исключая сам момент входа (чтобы не сработать на той же свече, если это не предусмотрено)
                    # Но для консервативности проверяем и текущую свечу (High/Low могли быть после Close)
                    # Для простоты и скорости берем срез [idx : next_signal_idx]
                    price_slice = df.loc[idx:next_signal_idx]
                    
                    exit_price_rm = None
                    exit_time_rm = None
                    exit_reason = None
                    
                    # Векторизированная проверка (быстрая) или итеративная (медленная)
                    # Для Trailing Stop нужна итерация или cummax
                    
                    if trailing_stop > 0:
                        # Trailing Stop Logic
                        # Рассчитываем динамический уровень стопа
                        highs = price_slice['high'].values
                        lows = price_slice['low'].values
                        times = price_slice.index
                        
                        # Максимальная цена с момента входа (накапливаемый максимум)
                        # Используем numpy для скорости
                        cum_max = np.maximum.accumulate(highs)
                        stop_levels = cum_max * (1 - trailing_stop)
                        
                        # Проверяем, где low пробил стоп
                        hit_indices = np.where(lows < stop_levels)[0]
                        
                        if len(hit_indices) > 0:
                            first_hit = hit_indices[0]
                            exit_time_rm = times[first_hit]
                            # Цена выхода - уровень стопа (или low, если проскользнули, берем стоп для идеальной модели)
                            exit_price_rm = float(stop_levels[first_hit])
                            exit_reason = 'trailing_stop'
                            
                            # Если также есть Take Profit, проверяем, не сработал ли он РАНЬШЕ
                            if take_profit > 0:
                                tp_price = entry_price * (1 + take_profit)
                                tp_hits = np.where(highs >= tp_price)[0]
                                if len(tp_hits) > 0:
                                    first_tp = tp_hits[0]
                                    if first_tp <= first_hit:
                                        exit_time_rm = times[first_tp]
                                        exit_price_rm = tp_price
                                        exit_reason = 'take_profit'
                    
                    elif take_profit > 0:
                        # Только Take Profit (без трейлинга)
                        tp_price = entry_price * (1 + take_profit)
                        # Ищем где high >= tp_price
                        tp_hits = price_slice[price_slice['high'] >= tp_price]
                        if not tp_hits.empty:
                            exit_time_rm = tp_hits.index[0]
                            exit_price_rm = tp_price
                            exit_reason = 'take_profit'
                    
                    # Если сработал Риск-менеджмент
                    if exit_price_rm is not None:
                        # Закрываем сделку
                        commission_buy = entry_price * commission_percent / 100
                        commission_sell = exit_price_rm * commission_percent / 100
                        total_commission = commission_buy + commission_sell
                        
                        profit = exit_price_rm - entry_price - total_commission
                        profit_percent = (profit / entry_price) * 100
                        
                        trades.append({
                            'entry_time': entry_time,
                            'exit_time': exit_time_rm,
                            'entry_price': entry_price,
                            'exit_price': exit_price_rm,
                            'commission': total_commission,
                            'profit': profit,
                            'profit_percent': profit_percent,
                            'reason': exit_reason
                        })
                        
                        entry_price = None
                        entry_time = None
                        
                        # ВАЖНО: Мы вышли по RM. 
                        # Если следующий сигнал был SELL (закрытие), мы его уже "опередили", нужно пропустить.
                        # Если следующий сигнал BUY, мы можем войти снова.
                        # Проверяем, какой сигнал был на next_signal_idx
                        # Если мы вышли РАНЬШЕ next_signal_idx, то всё ок, цикл продолжится.
                        # Но если next_signal_idx это SELL, который мы "заменили" выходом по стопу, 
                        # то в следующей итерации цикла (когда i увеличится), мы попадем на этот SELL.
                        # А так как entry_price is None, условие SELL не сработает. Всё корректно.
                        
                        # Единственный нюанс: если мы вышли ПОСЛЕ next_signal_idx (теоретически невозможно, т.к. срез до него)
                        # Или если выход совпал по времени.
                        
                        i += 1
                        continue

            # Продажа (Выход по стратегии)
            elif signal == -1 and entry_price is not None:
                exit_price = float(row['close'])
                
                commission_buy = entry_price * commission_percent / 100
                commission_sell = exit_price * commission_percent / 100
                total_commission = commission_buy + commission_sell
                
                profit = exit_price - entry_price - total_commission
                profit_percent = (profit / entry_price) * 100
                
                trades.append({
                    'entry_time': entry_time,
                    'exit_time': idx,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'commission': total_commission,
                    'profit': profit,
                    'profit_percent': profit_percent,
                    'reason': 'strategy'
                })
                
                entry_price = None
                entry_time = None
            
            i += 1
        
        # Если позиция открыта в конце, закрываем по последней цене
        if entry_price is not None:
            exit_price = df.iloc[-1]['close']
            
            # Учитываем комиссию: при покупке и при продаже
            commission_buy = entry_price * commission_percent / 100
            commission_sell = exit_price * commission_percent / 100
            total_commission = commission_buy + commission_sell
            
            # Прибыль с учетом комиссии
            profit = exit_price - entry_price - total_commission
            profit_percent = (profit / entry_price) * 100
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': df.index[-1],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'commission': total_commission,
                'profit': profit,
                'profit_percent': profit_percent
            })
        
        if not trades:
            pattern_name_str = ",".join(pattern_names) if pattern_names else pattern_type
            return StrategyResult(
                strategy_type=strategy_type,
                ema_short=ema_short,
                ema_long=ema_long,
                rsi_period=rsi_period,
                rsi_oversold=rsi_oversold,
                rsi_overbought=rsi_overbought,
                macd_fast=macd_fast,
                macd_slow=macd_slow,
                macd_signal=macd_signal,
                bb_period=bb_period,
                bb_std=bb_std,
                stoch_k=stoch_k,
                stoch_d=stoch_d,
                pattern_type=pattern_type,
                pattern_name=pattern_name_str,
                timeframe=timeframe,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                total_profit=0.0,
                profit_percent=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                win_rate=0.0
            )
        
        trades_df = pd.DataFrame(trades)
        
        # Расчет метрик
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['profit'] > 0])
        losing_trades = len(trades_df[trades_df['profit'] <= 0])
        total_profit = trades_df['profit'].sum()
        total_profit_percent = trades_df['profit_percent'].sum()
        
        # Максимальная просадка
        cumulative_profit = trades_df['profit'].cumsum()
        running_max = cumulative_profit.expanding().max()
        drawdown = cumulative_profit - running_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
        
        # Коэффициент Шарпа (упрощенный)
        if len(trades_df) > 1:
            returns = trades_df['profit_percent'].values
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252 * 24 * 60)  # Годовая нормализация
        else:
            sharpe_ratio = 0.0
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
        
        # Определяем название паттерна для сохранения
        pattern_name_str = ""
        if strategy_type == "PATTERN":
            if pattern_names:
                pattern_name_str = ",".join(pattern_names)
            else:
                pattern_name_str = pattern_type
        
        return StrategyResult(
            strategy_type=strategy_type,
            ema_short=ema_short,
            ema_long=ema_long,
            rsi_period=rsi_period,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
            bb_period=bb_period,
            bb_std=bb_std,
            stoch_k=stoch_k,
            stoch_d=stoch_d,
            pattern_type=pattern_type,
            pattern_name=pattern_name_str,
            timeframe=timeframe,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_profit=total_profit,
            profit_percent=total_profit_percent,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate
        )
    
    def optimize_strategy(
        self,
        df: pd.DataFrame,
        strategy_type: str = "EMA",
        ema_short_range: Tuple[int, int] = (5, 50),
        ema_long_range: Tuple[int, int] = (50, 200),
        rsi_period_range: Tuple[int, int] = (10, 20),
        rsi_oversold_range: Tuple[float, float] = (20.0, 40.0),
        rsi_overbought_range: Tuple[float, float] = (60.0, 80.0),
        macd_fast_range: Tuple[int, int] = (8, 16),
        macd_slow_range: Tuple[int, int] = (20, 30),
        macd_signal_range: Tuple[int, int] = (6, 12),
        bb_period_range: Tuple[int, int] = (15, 25),
        bb_std_range: Tuple[float, float] = (1.5, 2.5),
        stoch_k_range: Tuple[int, int] = (10, 18),
        stoch_d_range: Tuple[int, int] = (3, 5),
        pattern_type: str = "candlestick",
        pattern_names: List[str] = None,
        step: int = 5,
        timeframe: CandleInterval = CandleInterval.CANDLE_INTERVAL_1_MIN,
        commission_percent: float = 0.05,  # Комиссия в процентах за одну сделку (0.05% = 0.0005)
        progress_callback: Optional[callable] = None,
        take_profit: float = 0.0,
        trailing_stop: float = 0.0
    ) -> List[StrategyResult]:
        """Оптимизация стратегии путем перебора различных параметров"""
        interval_name = timeframe.name if hasattr(timeframe, 'name') else str(timeframe)
        logger.info(f"Начало оптимизации стратегии {strategy_type} на таймфрейме {interval_name}...")
        logger.info(f"Комиссия: {commission_percent}%, TP: {take_profit}, Trailing: {trailing_stop}")
        
        results = []
        
        # Helper to update progress
        def update_progress(current, total):
            if progress_callback:
                try:
                    progress_callback(current, total)
                except:
                    pass
            # Логируем в консоль реже
            if current % max(1, total // 10) == 0:
                logger.info(f"Прогресс: {current}/{total} комбинаций...")

        if strategy_type == "EMA":
            logger.info(f"Диапазон EMA Short: {ema_short_range[0]}-{ema_short_range[1]} (шаг {step})")
            logger.info(f"Диапазон EMA Long: {ema_long_range[0]}-{ema_long_range[1]} (шаг {step})")
            total_combinations = (
                ((ema_short_range[1] - ema_short_range[0]) // step + 1) *
                ((ema_long_range[1] - ema_long_range[0]) // step + 1)
            )
            current = 0
            
            for ema_short in range(ema_short_range[0], ema_short_range[1] + 1, step):
                for ema_long in range(ema_long_range[0], ema_long_range[1] + 1, step):
                    if ema_short >= ema_long:
                        continue
                    current += 1
                    update_progress(current, total_combinations)
                    try:
                        result = self.backtest_strategy(
                            df, strategy_type="EMA", ema_short=ema_short, ema_long=ema_long, 
                            timeframe=timeframe, commission_percent=commission_percent,
                            take_profit=take_profit, trailing_stop=trailing_stop
                        )
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"Ошибка при тестировании EMA({ema_short}, {ema_long}): {e}")
        
        elif strategy_type == "RSI":
            logger.info(f"Диапазон RSI Period: {rsi_period_range[0]}-{rsi_period_range[1]} (шаг {step})")
            total_combinations = (rsi_period_range[1] - rsi_period_range[0]) // step + 1
            current = 0
            
            for rsi_period in range(rsi_period_range[0], rsi_period_range[1] + 1, step):
                current += 1
                update_progress(current, total_combinations)
                try:
                    result = self.backtest_strategy(
                        df, strategy_type="RSI", rsi_period=rsi_period, 
                        rsi_oversold=30.0, rsi_overbought=70.0, 
                        timeframe=timeframe, commission_percent=commission_percent,
                        take_profit=take_profit, trailing_stop=trailing_stop
                    )
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Ошибка при тестировании RSI({rsi_period}): {e}")
        
        elif strategy_type == "MACD":
            logger.info(f"Диапазон MACD Fast: {macd_fast_range[0]}-{macd_fast_range[1]} (шаг {step})")
            logger.info(f"Диапазон MACD Slow: {macd_slow_range[0]}-{macd_slow_range[1]} (шаг {step})")
            total_combinations = (
                ((macd_fast_range[1] - macd_fast_range[0]) // step + 1) *
                ((macd_slow_range[1] - macd_slow_range[0]) // step + 1)
            )
            current = 0
            
            for macd_fast in range(macd_fast_range[0], macd_fast_range[1] + 1, step):
                for macd_slow in range(macd_slow_range[0], macd_slow_range[1] + 1, step):
                    if macd_fast >= macd_slow:
                        continue
                    current += 1
                    update_progress(current, total_combinations)
                    try:
                        result = self.backtest_strategy(
                            df, strategy_type="MACD", macd_fast=macd_fast, 
                            macd_slow=macd_slow, macd_signal=9, 
                            timeframe=timeframe, commission_percent=commission_percent,
                            take_profit=take_profit, trailing_stop=trailing_stop
                        )
                        results.append(result)
                    except Exception as e:
                        logger.warning(f"Ошибка при тестировании MACD({macd_fast}, {macd_slow}): {e}")
        
        elif strategy_type == "BB":
            logger.info(f"Диапазон BB Period: {bb_period_range[0]}-{bb_period_range[1]} (шаг {step})")
            total_combinations = (bb_period_range[1] - bb_period_range[0]) // step + 1
            current = 0
            
            for bb_period in range(bb_period_range[0], bb_period_range[1] + 1, step):
                current += 1
                update_progress(current, total_combinations)
                try:
                    result = self.backtest_strategy(
                        df, strategy_type="BB", bb_period=bb_period, bb_std=2.0, 
                        timeframe=timeframe, commission_percent=commission_percent,
                        take_profit=take_profit, trailing_stop=trailing_stop
                    )
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Ошибка при тестировании BB({bb_period}): {e}")
        
        elif strategy_type == "STOCH":
            logger.info(f"Диапазон Stochastic K: {stoch_k_range[0]}-{stoch_k_range[1]} (шаг {step})")
            total_combinations = (stoch_k_range[1] - stoch_k_range[0]) // step + 1
            current = 0
            
            for stoch_k in range(stoch_k_range[0], stoch_k_range[1] + 1, step):
                current += 1
                update_progress(current, total_combinations)
                try:
                    result = self.backtest_strategy(
                        df, strategy_type="STOCH", stoch_k=stoch_k, stoch_d=3, 
                        timeframe=timeframe, commission_percent=commission_percent,
                        take_profit=take_profit, trailing_stop=trailing_stop
                    )
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Ошибка при тестировании STOCH({stoch_k}): {e}")
        
        elif strategy_type == "PATTERN":
            logger.info(f"Оптимизация паттернов типа: {pattern_type}")
            if pattern_names:
                logger.info(f"Паттерны для тестирования: {', '.join(pattern_names)}")
            
            results_to_test = []
            if pattern_names is None or len(pattern_names) == 0:
                if pattern_type == "candlestick" or pattern_type == "combined":
                    results_to_test.extend(['hammer', 'inverted_hammer', 'bullish_engulfing', 'bearish_engulfing', 'shooting_star', 'hanging_man'])
                if pattern_type == "chart" or pattern_type == "combined":
                    results_to_test.extend(['double_bottom', 'double_top', 'triple_bottom', 'triple_top'])
            else:
                results_to_test.append(pattern_names)

            total_combinations = len(results_to_test)
            current = 0

            for pattern_item in results_to_test:
                current += 1
                update_progress(current, total_combinations)
                try:
                    # Если pattern_item это строка (одиночный паттерн) или список
                    p_names = [pattern_item] if isinstance(pattern_item, str) else pattern_item
                    
                    result = self.backtest_strategy(
                        df, strategy_type="PATTERN", pattern_type=pattern_type,
                        pattern_names=p_names, timeframe=timeframe, commission_percent=commission_percent,
                        take_profit=take_profit, trailing_stop=trailing_stop
                    )
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Ошибка при тестировании паттернов {pattern_item}: {e}")
        
        logger.info(f"Оптимизация завершена. Протестировано {len(results)} комбинаций")
        return results
    
    def find_best_strategy(self, results: List[StrategyResult], metric: str = "profit_percent") -> StrategyResult:
        """Поиск лучшей стратегии по указанной метрике"""
        if not results:
            raise ValueError("Нет результатов для анализа")
        
        # Фильтруем стратегии с хотя бы одной сделкой
        valid_results = [r for r in results if r.total_trades > 0]
        
        if not valid_results:
            raise ValueError("Нет стратегий с хотя бы одной сделкой")
        
        # Сортируем по выбранной метрике
        if metric == "profit_percent":
            best = max(valid_results, key=lambda x: x.profit_percent)
        elif metric == "sharpe_ratio":
            best = max(valid_results, key=lambda x: x.sharpe_ratio)
        elif metric == "win_rate":
            best = max(valid_results, key=lambda x: x.win_rate)
        else:
            best = max(valid_results, key=lambda x: x.profit_percent)
        
        return best
    
    def run_optimization(
        self,
        days: int = 30,
        ticker: str = "SBER",
        class_code: str = "TQBR",
        ema_short_range: Tuple[int, int] = (5, 50),
        ema_long_range: Tuple[int, int] = (50, 200),
        step: int = 5,
        timeframe: CandleInterval = CandleInterval.CANDLE_INTERVAL_1_MIN
    ) -> Tuple[pd.DataFrame, StrategyResult]:
        """Полный цикл оптимизации"""
        # Загрузка данных
        df = self.load_candle_data(days=days, ticker=ticker, class_code=class_code, interval=timeframe)
        
        # Оптимизация
        results = self.optimize_strategy(df, ema_short_range, ema_long_range, step, timeframe)
        
        # Поиск лучшей стратегии
        best = self.find_best_strategy(results, metric="profit_percent")
        
        # Конвертация результатов в DataFrame
        results_df = pd.DataFrame([
            {
                'strategy_type': r.strategy_type,
                'ema_short': r.ema_short,
                'ema_long': r.ema_long,
                'rsi_period': r.rsi_period,
                'macd_fast': r.macd_fast,
                'macd_slow': r.macd_slow,
                'bb_period': r.bb_period,
                'stoch_k': r.stoch_k,
                'pattern_type': r.pattern_type,
                'pattern_name': r.pattern_name,
                'timeframe': r.timeframe.name if hasattr(r.timeframe, 'name') else str(r.timeframe),
                'total_trades': r.total_trades,
                'winning_trades': r.winning_trades,
                'losing_trades': r.losing_trades,
                'total_profit': r.total_profit,
                'profit_percent': r.profit_percent,
                'max_drawdown': r.max_drawdown,
                'sharpe_ratio': r.sharpe_ratio,
                'win_rate': r.win_rate
            }
            for r in results
        ])
        
        return results_df, best


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    
    optimizer = StrategyOptimizer()
    
    print("Запуск оптимизации стратегии...")
    results_df, best = optimizer.run_optimization(
        days=30,
        ticker="SBER",
        ema_short_range=(5, 30),
        ema_long_range=(50, 150),
        step=5,
        timeframe=CandleInterval.CANDLE_INTERVAL_1_MIN
    )
    
    print("\n" + "="*80)
    print("ЛУЧШАЯ СТРАТЕГИЯ:")
    print("="*80)
    timeframe_name = best.timeframe.name if hasattr(best.timeframe, 'name') else str(best.timeframe)
    print(f"Таймфрейм: {timeframe_name}")
    print(f"EMA Short: {best.ema_short}")
    print(f"EMA Long: {best.ema_long}")
    print(f"Всего сделок: {best.total_trades}")
    print(f"Прибыльных: {best.winning_trades}")
    print(f"Убыточных: {best.losing_trades}")
    print(f"Винрейт: {best.win_rate:.2f}%")
    print(f"Общая прибыль: {best.total_profit:.2f} руб.")
    print(f"Общая прибыль: {best.profit_percent:.2f}%")
    print(f"Максимальная просадка: {best.max_drawdown:.2f} руб.")
    print(f"Коэффициент Шарпа: {best.sharpe_ratio:.2f}")
    print("="*80)
    
    # Топ-10 стратегий
    print("\nТОП-10 СТРАТЕГИЙ ПО ПРИБЫЛЬНОСТИ:")
    print("="*80)
    top_10 = results_df.nlargest(10, 'profit_percent')
    columns = ['ema_short', 'ema_long', 'timeframe', 'total_trades', 'profit_percent', 'win_rate', 'sharpe_ratio']
    # Фильтруем только существующие колонки
    columns = [c for c in columns if c in top_10.columns]
    print(top_10[columns].to_string())

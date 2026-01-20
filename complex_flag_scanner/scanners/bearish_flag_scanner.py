import os
import pandas as pd
import numpy as np
from datetime import timedelta
from t_tech.invest import Client, CandleInterval, InstrumentIdType, InstrumentStatus
from t_tech.invest.utils import now, quotation_to_decimal
from dotenv import load_dotenv
import sys
from pathlib import Path

# Добавляем путь к модулю neural_network для импорта функций проверки геометрии
sys.path.insert(0, str(Path(__file__).parent.parent))
from neural_network.check_annotations_geometry import check_short_constraints, check_lines_intersect_candles

load_dotenv()

class BearishFlagScanner:
    """
    Сканер для поиска паттерна Медвежий Флаг (шорт) со структурой 0-1-2-3-4:
    
    - T0: Начало падения (High перед флагштоком вниз)
    - T1: Низ флагштока (Low)
    - T2: Первый откат (High)
    - T3: Второй минимум (Low, >= T1)
    - T4: Второй откат (High, <= T2)
    - Сигнал: Генерируется при формировании T4 (паттерн готов к пробою)
    """
    
    def __init__(self, token: str):
        self.token = token
    
    def get_all_shares(self):
        """Получает список всех доступных акций Мосбиржи (TQBR)"""
        with Client(self.token) as client:
            response = client.instruments.shares(instrument_status=InstrumentStatus.INSTRUMENT_STATUS_BASE)
            shares = [
                s for s in response.instruments 
                if s.class_code == 'TQBR' and s.api_trade_available_flag and s.buy_available_flag
            ]
            return shares
        
    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет индикаторы EMA 7 и EMA 14"""
        if df.empty:
            return df
        df['ema_7'] = df['close'].ewm(span=7, adjust=False).mean()
        df['ema_14'] = df['close'].ewm(span=14, adjust=False).mean()
        return df

    def get_candles_df(self, ticker: str, class_code: str, days_back: int = 5, interval: CandleInterval = CandleInterval.CANDLE_INTERVAL_HOUR) -> pd.DataFrame:
        """Загружает свечи по Тикеру с указанным интервалом"""
        try:
            with Client(self.token) as client:
                item = client.instruments.get_instrument_by(
                    id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_TICKER,
                    class_code=class_code,
                    id=ticker
                ).instrument
                
                candles = client.get_all_candles(
                    instrument_id=item.uid,
                    from_=now() - timedelta(days=days_back),
                    to=now(),
                    interval=interval,
                )
                
                data = []
                for c in candles:
                    data.append({
                        'time': c.time,
                        'open': float(quotation_to_decimal(c.open)),
                        'high': float(quotation_to_decimal(c.high)),
                        'low': float(quotation_to_decimal(c.low)),
                        'close': float(quotation_to_decimal(c.close)),
                        'volume': c.volume
                    })
                    
            if not data:
                return pd.DataFrame()
                
            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['time'])
            # Конвертируем время в московский часовой пояс
            if df['time'].dt.tz is None:
                df['time'] = df['time'].dt.tz_localize('UTC')
            df['time'] = df['time'].dt.tz_convert('Europe/Moscow').dt.tz_localize(None)
            
            # Добавляем индикаторы
            df = self._add_indicators(df)
            
            return df
            
        except Exception as e:
            print(f"❌ Ошибка загрузки данных для {ticker} ({class_code}): {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def get_candles_df_by_dates(self, ticker: str, class_code: str, from_date, to_date, interval: CandleInterval = CandleInterval.CANDLE_INTERVAL_HOUR) -> pd.DataFrame:
        """Загружает свечи по Тикеру за указанный период с указанным интервалом"""
        try:
            with Client(self.token) as client:
                item = client.instruments.get_instrument_by(
                    id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_TICKER,
                    class_code=class_code,
                    id=ticker
                ).instrument
                
                candles = client.get_all_candles(
                    instrument_id=item.uid,
                    from_=from_date,
                    to=to_date,
                    interval=interval,
                )
                
                data = []
                for c in candles:
                    data.append({
                        'time': c.time,
                        'open': float(quotation_to_decimal(c.open)),
                        'high': float(quotation_to_decimal(c.high)),
                        'low': float(quotation_to_decimal(c.low)),
                        'close': float(quotation_to_decimal(c.close)),
                        'volume': c.volume
                    })
                    
            if not data:
                return pd.DataFrame()
                
            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['time'])
            # Конвертируем время в московский часовой пояс
            if df['time'].dt.tz is None:
                df['time'] = df['time'].dt.tz_localize('UTC')
            df['time'] = df['time'].dt.tz_convert('Europe/Moscow').dt.tz_localize(None)
            
            # Добавляем индикаторы
            df = self._add_indicators(df)
            
            return df
            
        except Exception as e:
            print(f"❌ Ошибка загрузки данных для {ticker} ({class_code}): {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def get_candles_by_uid(self, uid: str, days_back: int = 5, interval: CandleInterval = CandleInterval.CANDLE_INTERVAL_HOUR) -> pd.DataFrame:
        """Оптимизированный метод загрузки по UID с указанным интервалом"""
        try:
            with Client(self.token) as client:
                candles = client.get_all_candles(
                    instrument_id=uid,
                    from_=now() - timedelta(days=days_back),
                    to=now(),
                    interval=interval,
                )
                data = []
                for c in candles:
                    data.append({
                        'time': c.time,
                        'open': float(quotation_to_decimal(c.open)),
                        'high': float(quotation_to_decimal(c.high)),
                        'low': float(quotation_to_decimal(c.low)),
                        'close': float(quotation_to_decimal(c.close)),
                        'volume': c.volume
                    })
            if not data: return pd.DataFrame()
            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['time'])
            # Конвертируем время в московский часовой пояс
            if df['time'].dt.tz is None:
                df['time'] = df['time'].dt.tz_localize('UTC')
            df['time'] = df['time'].dt.tz_convert('Europe/Moscow').dt.tz_localize(None)
            
            # Добавляем индикаторы
            df = self._add_indicators(df)
            
            return df
        except Exception as e:
            print(f"❌ Ошибка загрузки данных по UID {uid}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def _find_local_extrema(self, df, window=3):
        """
        Находит локальные максимумы (highs) и минимумы (lows)
        window - количество соседних свечей для сравнения
        """
        highs_idx = []
        lows_idx = []
        
        for i in range(window, len(df) - window):
            # Максимум: high[i] больше всех соседей в окне
            window_highs = df.iloc[i-window:i+window+1]['high']
            if df.iloc[i]['high'] == window_highs.max():
                highs_idx.append(i)
            
            # Минимум: low[i] меньше всех соседей в окне
            window_lows = df.iloc[i-window:i+window+1]['low']
            if df.iloc[i]['low'] == window_lows.min():
                lows_idx.append(i)
        
        return np.array(highs_idx), np.array(lows_idx)

    def analyze(self, df: pd.DataFrame, debug=False, timeframe='1h', window=3, scan_type='all', min_pole_pct=None):
        """
        Основной метод анализа - ищет медвежий флаг
        
        Args:
            scan_type: 'latest' - ищет только последний актуальный паттерн (для сигналов),
                       'all' - ищет все паттерны в истории (для разметки)
            min_pole_pct: Минимальная высота флагштока в процентах (если None, берется дефолт по таймфрейму)
        """
        if len(df) < 50:
            return []
        
        return self.analyze_bearish_flag_0_1_2_3_4(df, debug=debug, timeframe=timeframe, window=window, scan_type=scan_type, min_pole_pct=min_pole_pct)

    def _calculate_quality(self, pattern):
        """
        Рассчитывает качество паттерна (0-100)
        Учитывает:
        1. Параллельность линий (50%)
        2. Наклон канала (вверх для медвежьего флага) (30%)
        3. Геометрию (T3 > T1) (20%)
        """
        t1 = pattern['t1']
        t3 = pattern['t3']
        t2 = pattern['t2']
        t4 = pattern['t4']
        
        if t3['idx'] == t1['idx'] or t4['idx'] == t2['idx']:
            return 30

        slope13 = (t3['price'] - t1['price']) / (t3['idx'] - t1['idx'])
        slope24 = (t4['price'] - t2['price']) / (t4['idx'] - t2['idx'])
        
        score = 0
        
        # 1. Параллельность (Max 50)
        avg_slope = (abs(slope13) + abs(slope24)) / 2
        if avg_slope == 0: avg_slope = 1e-6
        slope_diff_pct = abs(slope13 - slope24) / avg_slope
        parallel_score = 50 * max(0, 1 - slope_diff_pct)
        score += parallel_score
        
        # 2. Наклон канала (Max 30)
        # Для Медвежьего флага (SHORT) канал должен быть направлен ВВЕРХ (коррекция)
        if slope13 > 0 and slope24 > 0:
            score += 30
        elif slope13 > 0 or slope24 > 0:
            score += 15
            
        # 3. Геометрия (Max 20)
        # T3 должно быть выше T1 (четкая коррекция вверх)
        if t3['price'] > t1['price']:
            score += 10
        # T4 должно быть выше T3 (продолжение коррекции)
        if t4['price'] > t3['price']:
            score += 10
            
        return int(score)

    def _deduplicate_patterns(self, patterns):
        """
        Убирает дублирующиеся паттерны.
        """
        if not patterns:
            return []
            
        # Сортируем по качеству (лучшие первыми)
        patterns.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        
        unique_patterns = []
        
        for p in patterns:
            is_duplicate = False
            for up in unique_patterns:
                # Критерий дубликата: T1 и T4 находятся рядом
                t1_diff = abs(p['t1']['idx'] - up['t1']['idx'])
                t4_diff = abs(p['t4']['idx'] - up['t4']['idx'])
                
                if t1_diff < 5 and t4_diff < 5:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_patterns.append(p)
                
        return unique_patterns

    def analyze_bearish_flag_0_1_2_3_4(self, df: pd.DataFrame, debug=False, timeframe='1h', window=3, scan_type='all', min_pole_pct=None):
        """
        Ищет паттерн Медвежий Флаг (перевернутый) со структурой 0-1-2-3-4
        """
        if len(df) < 50:
            return []
        
        # Определяем минимальный процент флагштока если не задан
        if min_pole_pct is None:
            if '5m' in str(timeframe): min_pole_pct = 1.0
            elif '1h' in str(timeframe): min_pole_pct = 3.0
            elif '1d' in str(timeframe): min_pole_pct = 5.0
            else: min_pole_pct = 3.0
            
        found_patterns = []
        
        # 1. Находим локальные экстремумы
        highs_idx, lows_idx = self._find_local_extrema(df, window=window)
        
        if len(highs_idx) < 2 or len(lows_idx) < 2:
            return []
            
        # 2. Итерируемся по всем возможным T1 (дно флагштока)
        # T1 - это минимум
        
        for t1_i in range(len(lows_idx)):
            t1_idx = lows_idx[t1_i]
            t1_price = df.iloc[t1_idx]['low']
            
            # Ищем T3 (второе дно): любой минимум после T1
            for t3_i in range(t1_i + 1, len(lows_idx)):
                t3_idx = lows_idx[t3_i]
                
                if t3_idx - t1_idx > 60: 
                    break
                    
                t3_price = df.iloc[t3_idx]['low']
                
                # T3 не должен быть сильно ниже T1 (разрешаем -5% макс)
                if t3_price < t1_price * 0.95:
                    continue
                
                # Ищем T2: Глобальный максимум МЕЖДУ T1 и T3
                period_t1_t3 = df.iloc[t1_idx+1:t3_idx]
                if period_t1_t3.empty:
                    continue
                    
                t2_price = period_t1_t3['high'].max()
                t2_idx = period_t1_t3['high'].idxmax()
                
                # Ищем T4: Максимум ПОСЛЕ T3
                t4_candidates = [idx for idx in highs_idx if idx > t3_idx and idx - t3_idx < 30]
                
                for t4_idx in t4_candidates:
                    t4_price = df.iloc[t4_idx]['high']
                    
                    # Ищем T0: Максимум ПЕРЕД T1 (начало падения)
                    start_search = max(0, t1_idx - 50)
                    period_pre_t1 = df.iloc[start_search:t1_idx]
                    if period_pre_t1.empty:
                        continue
                    
                    t0_price = period_pre_t1['high'].max()
                    t0_idx = period_pre_t1['high'].idxmax()
                    
                    # Проверка на отсутствие промежуточных экстремумов НИЖЕ T1 между T0 и T1
                    # T1 должен быть самым низким дном на отрезке флагштока
                    t0_idx_int = int(t0_idx)
                    t1_idx_int = int(t1_idx)
                    
                    has_lower_low = False
                    
                    # Проверяем только lows
                    for l_idx in lows_idx:
                        if t0_idx_int < l_idx < t1_idx_int:
                            l_price = df.iloc[l_idx]['low']
                            # Если цена ниже или равна T1
                            if l_price <= t1_price:
                                has_lower_low = True
                                break
                    
                    if has_lower_low: continue
                    
                    # --- ВАЛИДАЦИЯ ---
                    
                    pole_height = t0_price - t1_price
                    if pole_height <= 0: continue
                    
                    # Проверка минимальной высоты флагштока в % (от вершины T0)
                    pole_pct = (pole_height / t0_price) * 100
                    if pole_pct < min_pole_pct:
                        continue
                    
                    violations = check_short_constraints(
                        t0_price, t1_price, t2_price, t3_price, t4_price, timeframe,
                        t0_idx, t1_idx, t2_idx, t3_idx, t4_idx
                    )
                    
                    if violations:
                        continue
                        
                    line_violations = check_lines_intersect_candles(
                        df, t1_idx, t1_price, t2_idx, t2_price, t3_idx, t3_price, t4_idx, t4_price, is_bullish=False
                    )
                    
                    if line_violations:
                        continue
                    
                    # Паттерн найден!
                    
                    pattern = {
                        'pattern': 'BEARISH_FLAG_0_1_2_3_4',
                        'timeframe': timeframe,
                        't0': {'idx': int(t0_idx), 'price': t0_price, 'time': df.iloc[t0_idx]['time']},
                        't1': {'idx': int(t1_idx), 'price': t1_price, 'time': df.iloc[t1_idx]['time']},
                        't2': {'idx': int(t2_idx), 'price': t2_price, 'time': df.iloc[t2_idx]['time']},
                        't3': {'idx': int(t3_idx), 'price': t3_price, 'time': df.iloc[t3_idx]['time']},
                        't4': {'idx': int(t4_idx), 'price': t4_price, 'time': df.iloc[t4_idx]['time']},
                        'pole_height': pole_height
                    }
                    
                    # Оценка качества
                    quality = self._calculate_quality(pattern)
                    pattern['quality_score'] = quality
                    
                    # Проверка свежести если нужно
                    current_idx = len(df) - 1
                    candles_after_t4 = current_idx - t4_idx
                    
                    if scan_type == 'latest':
                        if candles_after_t4 > 3:
                            continue
                    
                    found_patterns.append(pattern)
        
        # Фильтрация дубликатов
        found_patterns = self._deduplicate_patterns(found_patterns)
        
        return found_patterns


if __name__ == "__main__":
    # Тестирование
    TOKEN = os.environ.get("TINKOFF_INVEST_TOKEN")
    if not TOKEN:
        raise ValueError("Токен не найден!")
    
    scanner = BearishFlagScanner(TOKEN)
    
    # Тест на MXH6
    ticker = "MXH6"
    class_code = "SPBFUT"
    
    print(f"Загрузка данных для {ticker}...")
    df = scanner.get_candles_df(ticker, class_code, days_back=5)
    
    if df.empty:
        print("Нет данных")
    else:
        print(f"Загружено {len(df)} свечей")
        patterns = scanner.analyze(df, debug=True)
        
        if patterns:
            print("\nНайденные паттерны:")
            for p in patterns:
                print(f"  {p}")
        else:
            print("\nПаттерны не найдены")

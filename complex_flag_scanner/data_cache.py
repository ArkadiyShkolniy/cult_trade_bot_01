"""
Модуль для кэширования данных свечей в SQLite базе данных
Позволяет избежать повторных запросов к API для одних и тех же данных
"""
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
from t_tech.invest import CandleInterval


class DataCache:
    """
    Класс для кэширования данных свечей в SQLite базе данных
    """
    
    def __init__(self, cache_db_path: str = "data/candles_cache.db"):
        """
        Args:
            cache_db_path: Путь к файлу базы данных кэша
        """
        self.cache_db_path = Path(cache_db_path)
        self.cache_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Инициализирует базу данных и создает таблицу, если её нет"""
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS candles_cache (
                uid TEXT NOT NULL,
                interval TEXT NOT NULL,
                time TIMESTAMP NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (uid, interval, time)
            )
        """)
        
        # Создаем индексы для быстрого поиска
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_uid_interval 
            ON candles_cache(uid, interval)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_time 
            ON candles_cache(time)
        """)
        
        conn.commit()
        conn.close()
    
    def get_cached_candles(
        self, 
        uid: str, 
        interval: CandleInterval, 
        from_time: datetime, 
        to_time: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Получает свечи из кэша за указанный период
        
        Args:
            uid: UID инструмента
            interval: Интервал свечей
            from_time: Начальное время
            to_time: Конечное время
            
        Returns:
            DataFrame со свечами или None, если данных нет в кэше
        """
        conn = sqlite3.connect(self.cache_db_path)
        
        interval_str = self._interval_to_string(interval)
        
        query = """
            SELECT time, open, high, low, close, volume
            FROM candles_cache
            WHERE uid = ? AND interval = ? AND time >= ? AND time <= ?
            ORDER BY time ASC
        """
        
        df = pd.read_sql_query(
            query, 
            conn, 
            params=(uid, interval_str, from_time, to_time)
        )
        
        conn.close()
        
        if df.empty:
            return None
        
        df['time'] = pd.to_datetime(df['time'])
        return df
    
    def cache_candles(
        self, 
        uid: str, 
        interval: CandleInterval, 
        df: pd.DataFrame
    ):
        """
        Сохраняет свечи в кэш
        
        Args:
            uid: UID инструмента
            interval: Интервал свечей
            df: DataFrame со свечами (должен содержать колонки: time, open, high, low, close, volume)
        """
        if df.empty:
            return
        
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        interval_str = self._interval_to_string(interval)
        
        # Подготавливаем данные для вставки
        data = []
        for _, row in df.iterrows():
            data.append((
                uid,
                interval_str,
                row['time'],
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                int(row['volume']),
                datetime.now()
            ))
        
        # Используем INSERT OR REPLACE для обновления существующих записей
        cursor.executemany("""
            INSERT OR REPLACE INTO candles_cache 
            (uid, interval, time, open, high, low, close, volume, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)
        
        conn.commit()
        conn.close()
    
    def clear_cache(self, uid: Optional[str] = None, interval: Optional[CandleInterval] = None):
        """
        Очищает кэш
        
        Args:
            uid: UID инструмента (если None, очищает весь кэш)
            interval: Интервал свечей (если None, очищает все интервалы)
        """
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        if uid and interval:
            interval_str = self._interval_to_string(interval)
            cursor.execute(
                "DELETE FROM candles_cache WHERE uid = ? AND interval = ?",
                (uid, interval_str)
            )
        elif uid:
            cursor.execute("DELETE FROM candles_cache WHERE uid = ?", (uid,))
        elif interval:
            interval_str = self._interval_to_string(interval)
            cursor.execute("DELETE FROM candles_cache WHERE interval = ?", (interval_str,))
        else:
            cursor.execute("DELETE FROM candles_cache")
        
        conn.commit()
        conn.close()
    
    def get_cache_stats(self) -> dict:
        """
        Получает статистику кэша
        
        Returns:
            Словарь со статистикой
        """
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        # Общее количество записей
        cursor.execute("SELECT COUNT(*) FROM candles_cache")
        total_records = cursor.fetchone()[0]
        
        # Количество уникальных инструментов
        cursor.execute("SELECT COUNT(DISTINCT uid) FROM candles_cache")
        unique_uids = cursor.fetchone()[0]
        
        # Количество уникальных интервалов
        cursor.execute("SELECT COUNT(DISTINCT interval) FROM candles_cache")
        unique_intervals = cursor.fetchone()[0]
        
        # Время самой старой и новой записи
        cursor.execute("SELECT MIN(time), MAX(time) FROM candles_cache")
        time_range = cursor.fetchone()
        min_time = time_range[0] if time_range[0] else None
        max_time = time_range[1] if time_range[1] else None
        
        conn.close()
        
        return {
            'total_records': total_records,
            'unique_uids': unique_uids,
            'unique_intervals': unique_intervals,
            'min_time': min_time,
            'max_time': max_time
        }
    
    def _interval_to_string(self, interval: CandleInterval) -> str:
        """Преобразует CandleInterval в строку"""
        interval_map = {
            CandleInterval.CANDLE_INTERVAL_1_MIN: "1m",
            CandleInterval.CANDLE_INTERVAL_5_MIN: "5m",
            CandleInterval.CANDLE_INTERVAL_15_MIN: "15m",
            CandleInterval.CANDLE_INTERVAL_HOUR: "1h",
            CandleInterval.CANDLE_INTERVAL_DAY: "1d",
        }
        return interval_map.get(interval, str(interval))
    
    def _string_to_interval(self, interval_str: str) -> CandleInterval:
        """Преобразует строку в CandleInterval"""
        interval_map = {
            "1m": CandleInterval.CANDLE_INTERVAL_1_MIN,
            "5m": CandleInterval.CANDLE_INTERVAL_5_MIN,
            "15m": CandleInterval.CANDLE_INTERVAL_15_MIN,
            "1h": CandleInterval.CANDLE_INTERVAL_HOUR,
            "1d": CandleInterval.CANDLE_INTERVAL_DAY,
        }
        return interval_map.get(interval_str, CandleInterval.CANDLE_INTERVAL_HOUR)

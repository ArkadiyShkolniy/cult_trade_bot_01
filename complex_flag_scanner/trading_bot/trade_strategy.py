import pandas as pd
import numpy as np

class TradeStrategy:
    """
    Класс, отвечающий за логику принятия решений о входе в сделку.
    """

    def __init__(self):
        pass

    def calculate_line_price(self, start_idx, start_price, end_idx, end_price, current_idx):
        """
        Рассчитывает цену линии тренда в текущей точке (current_idx).
        """
        if end_idx == start_idx:
            return start_price
            
        slope = (end_price - start_price) / (end_idx - start_idx)
        return start_price + slope * (current_idx - start_idx)

    def check_entry_signal(self, df: pd.DataFrame, pattern: dict):
        """
        Проверяет условия входа "EMA Squeeze" (между EMA и линией тренда).
        
        Args:
            df: DataFrame со свечами (должен содержать 'ema_7', 'ema_14', 'close', 'open')
            pattern: Словарь с координатами паттерна
            
        Returns:
            bool: True если есть сигнал на вход, иначе False
            str: Описание причины (для логов)
        """
        # Получаем последнюю свечу
        current_candle = df.iloc[-1]
        current_idx = df.index[-1] # Или len(df)-1, зависит от индексации
        
        # Индикаторы
        ema7 = current_candle['ema_7']
        ema14 = current_candle['ema_14']
        close_price = current_candle['close']
        open_price = current_candle['open']
        
        # Координаты точек паттерна
        t1 = pattern['t1']
        t3 = pattern['t3']
        t2 = pattern['t2']
        t4 = pattern['t4']
        
        # Определяем направление паттерна
        # Если в названии есть BEARISH - это шорт, иначе лонг
        is_bullish = 'BEARISH' not in pattern.get('pattern', 'FLAG')
        
        if is_bullish:
            # --- ЛОГИКА LONG ---
            
            # 1. Проверка тренда (EMA 7 выше EMA 14)
            if ema7 <= ema14:
                return False, "Нет тренда (EMA 7 <= EMA 14)"
                
            # 2. Проверка положения цены относительно EMA
            # Цена должна быть выше EMA 7 (сильный тренд) или хотя бы выше EMA 14
            # Строгий вариант: Close > EMA 7
            if close_price < ema7:
                return False, "Цена ниже EMA 7"
                
            # 3. Расчет линии сопротивления (T1-T3)
            # Нам нужно спроецировать линию на текущую свечу
            # ВАЖНО: индексы в df должны совпадать с индексами паттерна
            t1_idx = t1['idx']
            t3_idx = t3['idx']
            
            # Текущий индекс относительно начала df (если паттерн найден на истории)
            # Если мы торгуем в реальном времени, current_idx это последняя свеча
            # Предполагаем, что current_idx > t4_idx
            
            line_1_3_price = self.calculate_line_price(
                t1_idx, t1['price'], 
                t3_idx, t3['price'], 
                len(df) - 1 # Индекс последней свечи
            )
            
            # 4. Проверка: Цена ниже линии 1-3? (Мы входим ВНУТРИ треугольника)
            if close_price > line_1_3_price:
                return False, "Цена уже пробила линию 1-3 (поздно для Squeeze входа)"
                
            # 5. Триггер: Зеленая свеча (подтверждение отскока)
            if close_price <= open_price:
                return False, "Свеча не зеленая"
                
            return True, f"SIGNAL LONG: Price {close_price:.2f} зажата между EMA7 ({ema7:.2f}) и Линией 1-3 ({line_1_3_price:.2f})"

        else:
            # --- ЛОГИКА SHORT ---
            
            # 1. Проверка тренда (EMA 7 ниже EMA 14)
            if ema7 >= ema14:
                return False, "Нет тренда (EMA 7 >= EMA 14)"
                
            # 2. Проверка цены (Ниже EMA 7)
            if close_price > ema7:
                return False, "Цена выше EMA 7"
                
            # 3. Линия поддержки (T1-T3 для шорта это нижняя линия)
            t1_idx = t1['idx']
            t3_idx = t3['idx']
            
            line_1_3_price = self.calculate_line_price(
                t1_idx, t1['price'], 
                t3_idx, t3['price'], 
                len(df) - 1
            )
            
            # 4. Проверка: Цена выше линии 1-3? (Внутри флага)
            if close_price < line_1_3_price:
                return False, "Цена уже пробила линию 1-3 вниз"
                
            # 5. Триггер: Красная свеча
            if close_price >= open_price:
                return False, "Свеча не красная"
                
            return True, f"SIGNAL SHORT: Price {close_price:.2f} зажата между EMA7 ({ema7:.2f}) и Линией 1-3 ({line_1_3_price:.2f})"

    def calculate_exit_levels(self, df: pd.DataFrame, pattern: dict, entry_price: float):
        """
        Рассчитывает уровни Stop Loss и Take Profit.
        
        Args:
            df: DataFrame со свечами (нужен для EMA)
            pattern: Словарь паттерна (нужен для T0/T1)
            entry_price: Цена входа
            
        Returns:
            dict: {'stop_loss': float, 'take_profit': float}
        """
        # Направление сделки
        is_bullish = 'BEARISH' not in pattern.get('pattern', 'FLAG')
        
        # EMA14 текущей свечи для стопа
        current_ema14 = df.iloc[-1]['ema_14']
        
        if is_bullish:
            # --- LONG ---
            # Stop Loss: За EMA 14 (чуть ниже, например на 0.1%)
            stop_loss = current_ema14 * 0.999
            
            # Take Profit: Цель - T1 (вершина флагштока для Long)
            # В нотации пользователя T0 -> T1 это движение флагштока.
            # Для Long: T0 - минимум, T1 - максимум. Цель - вернуться к T1.
            take_profit = pattern['t1']['price']
            
            return {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'desc': f"SL: EMA14-0.1% ({stop_loss:.2f}), TP: T1 ({take_profit:.2f})"
            }
            
        else:
            # --- SHORT ---
            # Stop Loss: За EMA 14 (чуть выше, на 0.1%)
            stop_loss = current_ema14 * 1.001
            
            # Take Profit: Цель - T1 (дно флагштока)
            take_profit = pattern['t1']['price']
            
            return {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'desc': f"SL: EMA14+0.1% ({stop_loss:.2f}), TP: T1 ({take_profit:.2f})"
            }


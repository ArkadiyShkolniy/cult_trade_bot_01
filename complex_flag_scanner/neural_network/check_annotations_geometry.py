#!/usr/bin/env python3
"""
Проверка размеченных данных на соответствие новым геометрическим ограничениям
"""

import pandas as pd
import numpy as np
from pathlib import Path


def check_channel_boundary_violation(candles_df, start_idx, start_price, end_idx, end_price, is_upper_boundary):
    """
    Проверяет, нарушают ли свечи границу канала (линию).
    is_upper_boundary=True: линия сверху, свечи не должны закрываться выше нее.
    is_upper_boundary=False: линия снизу, свечи не должны закрываться ниже нее.
    """
    if start_idx == end_idx:
        return False
    
    slope = (end_price - start_price) / (end_idx - start_idx)
    
    # Индексы для итерации (строго между точками)
    s_idx = min(start_idx, end_idx)
    e_idx = max(start_idx, end_idx)
    
    for idx in range(s_idx + 1, e_idx):
        if idx >= len(candles_df):
            continue
        
        candle = candles_df.iloc[idx]
        line_price = start_price + slope * (idx - start_idx)
        
        body_top = max(candle['open'], candle['close'])
        body_bottom = min(candle['open'], candle['close'])
        
        # Толеранс для касания (можно чуть-чуть задеть, но не сильно)
        # Используем 0.05% от цены как допустимую погрешность касания
        tolerance = line_price * 0.0005
        
        if is_upper_boundary:
            # Линия сверху (сопротивление). Тело не должно быть значимо выше.
            # Нарушение если body_top > line_price + tolerance
            if body_top > line_price + tolerance:
                return True
        else:
            # Линия снизу (поддержка). Тело не должно быть значимо ниже.
            # Нарушение если body_bottom < line_price - tolerance
            if body_bottom < line_price - tolerance:
                return True
                
    return False


def check_lines_intersect_candles(candles_df, t1_idx, t1_price, t2_idx, t2_price, 
                                   t3_idx, t3_price, t4_idx, t4_price, is_bullish=True):
    """
    Проверяет, что линии канала не нарушаются свечами.
    """
    violations = []
    
    # Определяем роль линий в зависимости от типа паттерна
    if is_bullish: # LONG
        # T1-T3 соединяет Highs -> Верхняя граница
        is_13_upper = True
        # T2-T4 соединяет Lows -> Нижняя граница
        is_24_upper = False
    else: # SHORT
        # T1-T3 соединяет Lows -> Нижняя граница
        is_13_upper = False
        # T2-T4 соединяет Highs -> Верхняя граница
        is_24_upper = True
    
    # 1. Линия T1-T3
    if check_channel_boundary_violation(candles_df, t1_idx, t1_price, t3_idx, t3_price, is_13_upper):
        boundary_type = "верхнюю" if is_13_upper else "нижнюю"
        violations.append(f"Свечи пробивают {boundary_type} линию T1-T3 между {t1_idx} и {t3_idx}")
    
    # 2. Линия T2-T4
    if check_channel_boundary_violation(candles_df, t2_idx, t2_price, t4_idx, t4_price, is_24_upper):
        boundary_type = "верхнюю" if is_24_upper else "нижнюю"
        violations.append(f"Свечи пробивают {boundary_type} линию T2-T4 между {t2_idx} и {t4_idx}")

    # 3. Проверка продления линии T1-T3 до T4
    if t3_idx != t1_idx and t4_idx > t3_idx:
        slope_13 = (t3_price - t1_price) / (t3_idx - t1_idx)
        projected_price_at_t4 = t1_price + slope_13 * (t4_idx - t1_idx)
        
        if check_channel_boundary_violation(candles_df, t3_idx, t3_price, t4_idx, projected_price_at_t4, is_13_upper):
             boundary_type = "верхнюю" if is_13_upper else "нижнюю"
             violations.append(f"Свечи пробивают продолжение {boundary_type} линии T1-T3 на участке T3-T4")
    
    return violations

def get_tolerance_percent(timeframe):
    """
    Возвращает процент погрешности в зависимости от таймфрейма.
    5m - 0.1%
    1h - 0.3%
    1d - 0.5%
    """
    tf = str(timeframe).lower()
    
    # Точные совпадения для основных таймфреймов
    if '5m' in tf:
        return 0.001  # 0.1%
    elif '1h' in tf:
        return 0.003  # 0.3%
    elif '1d' in tf:
        return 0.005  # 0.5%
        
    # Логика для остальных таймфреймов
    elif 'm' in tf: # Минутные (10m, 15m и т.д.)
        return 0.001
    elif 'h' in tf: # Часовые (4h и т.д.)
        return 0.003
    elif 'd' in tf or 'w' in tf: # Дневные и недельные
        return 0.005
    else:
        return 0.003


def check_long_constraints(T0, T1, T2, T3, T4, timeframe='1h', t0_idx=None, t1_idx=None, t2_idx=None, t3_idx=None, t4_idx=None):
    """
    Проверяет соблюдение ограничений для LONG
    """
    violations = []
    tolerance_percent = get_tolerance_percent(timeframe)
    
    # 1. T2 >= T1 - 0.62 * (T1 - T0)
    min_t2 = T1 - 0.62 * (T1 - T0)
    tolerance_t2 = min_t2 * tolerance_percent
    if T2 < min_t2 - tolerance_t2:
        violations.append(f"T2 ({T2:.2f}) < фиба 0.62 ({min_t2:.2f})")
    
    # 2. T3: диапазон фиба 0.5 хода T1-T2, но не выше T1
    move_12 = T1 - T2
    fib_50_level = T2 + 0.5 * move_12
    tolerance_t3 = fib_50_level * tolerance_percent
    
    if T3 < fib_50_level - tolerance_t3:
        violations.append(f"T3 ({T3:.2f}) < фиба 0.5 ({fib_50_level:.2f})")
    
    # STRICT CHECK: T3 <= T1 (с учетом толерантности)
    # T3 может быть чуть выше T1 в пределах погрешности
    tolerance_t1 = T1 * tolerance_percent
    if T3 > T1 + tolerance_t1:
        violations.append(f"T3 ({T3:.2f}) > T1 ({T1:.2f}) + погрешность {tolerance_t1:.2f} ({tolerance_percent*100}%)")
    
    # 3. T4 constraints
    move_23 = T3 - T2
    max_t4_from_t3 = T3 - 0.5 * move_23
    min_t4_from_pole = T1 - 0.62 * (T1 - T0)
    tolerance_t4 = max_t4_from_t3 * tolerance_percent
    
    if T4 > max_t4_from_t3 + tolerance_t4:
        violations.append(f"T4 ({T4:.2f}) > max отката ({max_t4_from_t3:.2f})")
    
    if T4 < min_t4_from_pole - (min_t4_from_pole * tolerance_percent):
        violations.append(f"T4 ({T4:.2f}) < фиба 0.62 ({min_t4_from_pole:.2f})")
    
    # 4. Проверка линий (Divergence & Slope)
    if all(x is not None for x in [t1_idx, t2_idx, t3_idx, t4_idx]):
        if (t3_idx - t1_idx) != 0 and (t4_idx - t2_idx) != 0:
            slope_13 = (T3 - T1) / (t3_idx - t1_idx) # Верхняя линия
            slope_24 = (T4 - T2) / (t4_idx - t2_idx) # Нижняя линия
            
            # Расхождение: Top (13) slope > Bottom (24) slope
            # В бычьем флаге (канал вниз) slope < 0.
            # Если 13 > 24 (например -0.5 > -1.0), то это расхождение.
            # STRICT CHECK: Разрешаем равенство или схождение.
            if slope_13 > slope_24:
                violations.append(f"Линии расходятся (slope_top={slope_13:.4f} > slope_bottom={slope_24:.4f})")
            
            # Проверка пересечения линий (не должны пересекаться до T4)
            # Вычисляем точку пересечения
            if abs(slope_13 - slope_24) > 1e-6: # Если не параллельны
                # x_int = (c2 - c1) / (m1 - m2)
                # y = m*x + c => c = y - m*x
                c1 = T1 - slope_13 * t1_idx
                c2 = T2 - slope_24 * t2_idx
                x_int = (c2 - c1) / (slope_13 - slope_24)
                
                # Если пересечение происходит внутри паттерна (от T1 до T4 включительно)
                # Если x_int чуть меньше T1 (расхождение), это ловится проверкой slope
                # Нас волнует схождение слишком рано
                if t1_idx < x_int <= t4_idx:
                     violations.append(f"Линии пересекаются слишком рано (idx={x_int:.1f} <= T4={t4_idx})")

            # Наклон канала: должен быть вниз (slope < 0) или горизонтально (~0)
            # Запрещаем явный наклон вверх
            if slope_13 > 0.05 and slope_24 > 0.05:
                violations.append(f"Канал флага направлен вверх, ожидается вниз")
    
    return violations


def check_short_constraints(T0, T1, T2, T3, T4, timeframe='1h', t0_idx=None, t1_idx=None, t2_idx=None, t3_idx=None, t4_idx=None):
    """
    Проверяет соблюдение ограничений для SHORT
    """
    violations = []
    tolerance_percent = get_tolerance_percent(timeframe)
    
    # 1. T2 <= T1 + 0.62 * (T0 - T1)
    max_t2 = T1 + 0.62 * (T0 - T1)
    tolerance_t2 = max_t2 * tolerance_percent
    if T2 > max_t2 + tolerance_t2:
        violations.append(f"T2 ({T2:.2f}) > фиба 0.62 ({max_t2:.2f})")
    
    # 2. T3: диапазон фиба 0.5, но не ниже T1
    move_12 = T2 - T1
    fib_50_level = T1 + 0.5 * move_12
    tolerance_t3 = fib_50_level * tolerance_percent
    
    # STRICT CHECK: T3 >= T1 (с учетом толерантности)
    tolerance_t1 = T1 * tolerance_percent
    if T3 < T1 - tolerance_t1:
        violations.append(f"T3 ({T3:.2f}) < T1 ({T1:.2f}) - погрешность {tolerance_t1:.2f} ({tolerance_percent*100}%)")
    
    if T3 > fib_50_level + tolerance_t3:
        violations.append(f"T3 ({T3:.2f}) > фиба 0.5 ({fib_50_level:.2f})")
    
    # 3. T4 constraints
    move_23 = T2 - T3
    min_t4_from_t3 = T3 + 0.5 * move_23
    max_t4_from_pole = T1 + 0.62 * (T0 - T1)
    tolerance_t4 = min_t4_from_t3 * tolerance_percent
    
    if T4 < min_t4_from_t3 - tolerance_t4:
        violations.append(f"T4 ({T4:.2f}) < min отката ({min_t4_from_t3:.2f})")
    
    if T4 > max_t4_from_pole + (max_t4_from_pole * tolerance_percent):
        violations.append(f"T4 ({T4:.2f}) > фиба 0.62 ({max_t4_from_pole:.2f})")
    
    # 4. Проверка линий (Divergence & Slope)
    if all(x is not None for x in [t1_idx, t2_idx, t3_idx, t4_idx]):
        if (t3_idx - t1_idx) != 0 and (t4_idx - t2_idx) != 0:
            slope_13 = (T3 - T1) / (t3_idx - t1_idx) # Нижняя линия (для SHORT T1, T3 внизу)
            slope_24 = (T4 - T2) / (t4_idx - t2_idx) # Верхняя линия (для SHORT T2, T4 вверху)
            
            # Расхождение для SHORT (канал вверх).
            # Расхождение если Top (24) растет быстрее Bottom (13).
            # slope_24 > slope_13.
            if slope_24 > slope_13:
                violations.append(f"Линии расходятся (slope_top={slope_24:.4f} > slope_bottom={slope_13:.4f})")
            
            # Проверка пересечения линий (не должны пересекаться до T4)
            if abs(slope_13 - slope_24) > 1e-6:
                c1 = T1 - slope_13 * t1_idx
                c2 = T2 - slope_24 * t2_idx
                x_int = (c2 - c1) / (slope_13 - slope_24)
                
                if t1_idx < x_int <= t4_idx:
                     violations.append(f"Линии пересекаются слишком рано (idx={x_int:.1f} <= T4={t4_idx})")

            # Наклон канала: должен быть вверх (slope > 0) или горизонтально
            # Запрещаем явный наклон вниз (slope < -0.05)
            if slope_13 < -0.05 and slope_24 < -0.05:
                violations.append(f"Канал флага направлен вниз, ожидается вверх")
    
    return violations


def main():
    # ... (код main остается тем же, нужен для ручного запуска)
    pass

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Скрипт для добавления координат точек T0-T4 в уже размеченные паттерны

Варианты:
1. Если есть JSON файлы с координатами - использует их
2. Если нет - можно использовать математический сканер для автоматического определения
3. Или предложить переразметку через labeling_dashboard
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Добавляем путь к корню проекта
sys.path.insert(0, str(Path(__file__).parent.parent))

from neural_network.annotator import PatternAnnotator
from scanners.combined_scanner import ComplexFlagScanner
from dotenv import load_dotenv

load_dotenv()


def load_json_metadata(file_path):
    """Загружает метаданные из JSON файла"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except:
        return None


def find_pattern_with_scanner(df, label, scanner, timeframe='1h'):
    """
    Пытается найти паттерн в данных с помощью математического сканера
    
    Args:
        df: DataFrame со свечами
        label: Метка (1=бычий, 2=медвежий)
        scanner: ComplexFlagScanner
        timeframe: Таймфрейм
    
    Returns:
        dict с координатами точек или None
    """
    try:
        # Ищем паттерны
        patterns = scanner.analyze(df, timeframe=timeframe)
        
        if not patterns:
            return None
        
        # Фильтруем по типу (бычий/медвежий)
        for pattern_info in patterns:
            is_bearish = "BEARISH" in pattern_info.get('pattern', '')
            pattern_label = 2 if is_bearish else 1
            
            if pattern_label == label:
                # Найден подходящий паттерн
                return {
                    'T0': pattern_info.get('t0'),
                    'T1': pattern_info.get('t1'),
                    'T2': pattern_info.get('t2'),
                    'T3': pattern_info.get('t3'),
                    'T4': pattern_info.get('t4')
                }
        
        return None
    except Exception as e:
        print(f"   ⚠️  Ошибка при поиске паттерна: {e}")
        return None


def add_coordinates_from_json(annotations_df, data_dir):
    """
    Добавляет координаты из JSON файлов с метаданными
    """
    data_dir = Path(data_dir)
    candles_dir = data_dir / 'candles'
    
    updated_count = 0
    
    for idx, row in annotations_df.iterrows():
        # Пропускаем, если координаты уже есть
        if pd.notna(row.get('t0_idx')):
            continue
        
        file_path = data_dir / row['file']
        json_path = file_path.with_suffix('.json')
        
        if json_path.exists():
            metadata = load_json_metadata(json_path)
            if metadata and 't0' in metadata:
                # Извлекаем координаты из JSON
                for point_name in ['T0', 'T1', 'T2', 'T3', 'T4']:
                    point_lower = point_name.lower()
                    point_data = metadata.get(point_lower)
                    
                    if point_data and isinstance(point_data, dict):
                        annotations_df.at[idx, f'{point_lower}_idx'] = point_data.get('idx')
                        annotations_df.at[idx, f'{point_lower}_price'] = point_data.get('price')
                
                updated_count += 1
                print(f"   ✅ {row['file']}: координаты добавлены из JSON")
    
    return annotations_df, updated_count


def add_coordinates_from_scanner(annotations_df, data_dir, use_scanner=False):
    """
    Добавляет координаты используя математический сканер (опционально)
    """
    if not use_scanner:
        return annotations_df, 0
    
    token = os.environ.get("TINKOFF_INVEST_TOKEN")
    if not token:
        print("   ⚠️  Токен не найден, пропускаем сканер")
        return annotations_df, 0
    
    scanner = ComplexFlagScanner(token)
    data_dir = Path(data_dir)
    updated_count = 0
    
    for idx, row in annotations_df.iterrows():
        # Пропускаем, если координаты уже есть
        if pd.notna(row.get('t0_idx')):
            continue
        
        # Пропускаем label=0 (нет паттерна)
        label = row.get('label', 0)
        if label == 0:
            continue
        
        file_path = data_dir / row['file']
        if not file_path.exists():
            continue
        
        try:
            # Загружаем свечи
            df = pd.read_csv(file_path)
            timeframe = row.get('timeframe', '1h')
            
            # Ищем паттерн
            points = find_pattern_with_scanner(df, label, scanner, timeframe)
            
            if points:
                # Добавляем координаты
                for point_name in ['T0', 'T1', 'T2', 'T3', 'T4']:
                    point_lower = point_name.lower()
                    point_data = points.get(point_name)
                    
                    if point_data and isinstance(point_data, dict):
                        annotations_df.at[idx, f'{point_lower}_idx'] = point_data.get('idx')
                        annotations_df.at[idx, f'{point_lower}_price'] = point_data.get('price')
                
                updated_count += 1
                print(f"   ✅ {row['file']}: координаты найдены сканером")
            else:
                print(f"   ⚠️  {row['file']}: паттерн не найден сканером")
        
        except Exception as e:
            print(f"   ❌ {row['file']}: ошибка - {e}")
    
    return annotations_df, updated_count


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Добавление координат в уже размеченные паттерны')
    parser.add_argument('--data_dir', type=str, default='neural_network/data',
                        help='Директория с данными')
    parser.add_argument('--use_scanner', action='store_true',
                        help='Использовать математический сканер для автоматического определения координат')
    parser.add_argument('--dry_run', action='store_true',
                        help='Только показать что будет сделано, не сохранять')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    annotations_file = data_dir / 'annotations.csv'
    
    if not annotations_file.exists():
        print(f"❌ Файл аннотаций не найден: {annotations_file}")
        return
    
    print("=" * 60)
    print("🔧 ДОБАВЛЕНИЕ КООРДИНАТ В РАЗМЕЧЕННЫЕ ПАТТЕРНЫ")
    print("=" * 60)
    print()
    
    # Загружаем аннотации
    annotations_df = pd.read_csv(annotations_file)
    print(f"📊 Загружено аннотаций: {len(annotations_df)}")
    
    # Проверяем, есть ли колонки с координатами
    coord_cols = [f'{p}_idx' for p in ['t0', 't1', 't2', 't3', 't4']]
    if not all(col in annotations_df.columns for col in coord_cols):
        print("   ⚠️  Колонки с координатами отсутствуют, добавляем...")
        for col in coord_cols + [col.replace('_idx', '_price') for col in coord_cols]:
            if col not in annotations_df.columns:
                annotations_df[col] = None
    
    # Подсчитываем статистику
    has_coords = annotations_df['t0_idx'].notna() if 't0_idx' in annotations_df.columns else pd.Series([False]*len(annotations_df))
    print(f"   ✅ С координатами: {has_coords.sum()}")
    print(f"   ❌ Без координат: {(~has_coords).sum()}")
    print()
    
    if has_coords.sum() == len(annotations_df):
        print("✅ Все аннотации уже имеют координаты!")
        return
    
    # Попытка 1: Добавляем из JSON файлов
    print("📋 Попытка 1: Загрузка координат из JSON файлов...")
    annotations_df, json_count = add_coordinates_from_json(annotations_df, data_dir)
    print(f"   ✅ Добавлено из JSON: {json_count}")
    print()
    
    # Попытка 2: Используем сканер (если указан флаг)
    if args.use_scanner:
        print("📋 Попытка 2: Поиск координат с помощью математического сканера...")
        annotations_df, scanner_count = add_coordinates_from_scanner(annotations_df, data_dir, use_scanner=True)
        print(f"   ✅ Добавлено сканером: {scanner_count}")
        print()
    else:
        scanner_count = 0
    
    # Итоговая статистика
    has_coords_after = annotations_df['t0_idx'].notna()
    still_missing = (~has_coords_after).sum()
    
    print("=" * 60)
    print("📊 ИТОГИ")
    print("=" * 60)
    print(f"   Добавлено из JSON: {json_count}")
    print(f"   Добавлено сканером: {scanner_count}")
    print(f"   Всего с координатами: {has_coords_after.sum()}")
    print(f"   Все еще без координат: {still_missing}")
    print()
    
    if still_missing > 0:
        print("⚠️  Для оставшихся паттернов нужно:")
        print("   1. Переразметить через labeling_dashboard")
        print("   2. Или добавить JSON файлы с координатами")
        print()
    
    # Сохраняем (если не dry_run)
    if not args.dry_run:
        annotations_df.to_csv(annotations_file, index=False)
        print(f"✅ Аннотации сохранены: {annotations_file}")
    else:
        print("🔍 DRY RUN: изменения не сохранены (используйте без --dry_run для сохранения)")
    
    print("=" * 60)


if __name__ == "__main__":
    main()


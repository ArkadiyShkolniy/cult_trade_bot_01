"""
Модуль для разметки паттернов (аннотации данных)
Интерфейс для маркировки найденных паттернов сканером
"""

import os
import pandas as pd
import json
from datetime import datetime
from pathlib import Path


class PatternAnnotator:
    """
    Класс для разметки паттернов найденных сканером
    """
    
    def __init__(self, data_dir='neural_network/data', annotations_file='annotations.csv'):
        """
        Args:
            data_dir: Директория для хранения данных
            annotations_file: Имя файла с аннотациями
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.annotations_file = self.data_dir / annotations_file
        self.candles_dir = self.data_dir / 'candles'
        self.candles_dir.mkdir(exist_ok=True)
        
        # Загружаем существующие аннотации
        if self.annotations_file.exists():
            self.annotations = pd.read_csv(self.annotations_file)
        else:
            self.annotations = pd.DataFrame(columns=[
                'file', 'label', 'ticker', 'timeframe', 
                'pattern_type', 'timestamp', 'notes',
                't0_idx', 't0_price', 't1_idx', 't1_price',
                't2_idx', 't2_price', 't3_idx', 't3_price',
                't4_idx', 't4_price'
            ])
    
    def save_candles(self, df, ticker, timeframe, pattern_info=None):
        """
        Сохраняет свечи в CSV файл
        
        Args:
            df: DataFrame со свечами
            ticker: Тикер инструмента
            timeframe: Таймфрейм
            pattern_info: Информация о найденном паттерне (опционально)
        
        Returns:
            Путь к сохраненному файлу
        """
        # Генерируем имя файла
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{ticker}_{timeframe}_{timestamp}.csv"
        filepath = self.candles_dir / filename
        
        # Сохраняем свечи
        df.to_csv(filepath, index=False)
        
        # Если есть информация о паттерне, сохраняем метаданные
        if pattern_info:
            metadata_file = filepath.with_suffix('.json')
            with open(metadata_file, 'w') as f:
                json.dump(pattern_info, f, indent=2, default=str)
        
        return str(filepath.relative_to(self.data_dir))
    
    def annotate_pattern(self, candles_file, label, ticker, timeframe, 
                        pattern_type=None, notes=None, points=None):
        """
        Добавляет аннотацию для паттерна
        
        Args:
            candles_file: Относительный путь к файлу со свечами
            label: Метка класса (0=нет паттерна, 1=бычий, 2=медвежий)
            ticker: Тикер инструмента
            timeframe: Таймфрейм
            pattern_type: Тип паттерна (опционально)
            notes: Заметки (опционально)
            points: Словарь с координатами точек T0-T4 (опционально)
                   Формат: {'T0': {'idx': int, 'price': float}, ...}
        """
        annotation = {
            'file': candles_file,
            'label': int(label),
            'ticker': ticker,
            'timeframe': timeframe,
            'pattern_type': pattern_type or '',
            'timestamp': datetime.now().isoformat(),
            'notes': notes or '',
            't0_idx': None, 't0_price': None,
            't1_idx': None, 't1_price': None,
            't2_idx': None, 't2_price': None,
            't3_idx': None, 't3_price': None,
            't4_idx': None, 't4_price': None
        }
        
        # Добавляем координаты точек, если они переданы
        if points:
            for point_name in ['T0', 'T1', 'T2', 'T3', 'T4']:
                if point_name in points and points[point_name]:
                    point_data = points[point_name]
                    if isinstance(point_data, dict) and 'idx' in point_data and 'price' in point_data:
                        annotation[f'{point_name.lower()}_idx'] = int(point_data['idx'])
                        annotation[f'{point_name.lower()}_price'] = float(point_data['price'])
        
        # Добавляем к аннотациям
        self.annotations = pd.concat([
            self.annotations,
            pd.DataFrame([annotation])
        ], ignore_index=True)
        
        # Сохраняем
        self.save_annotations()
        
        print(f"✅ Аннотация добавлена: {candles_file} (label={label})")
    
    def delete_annotation(self, file_path):
        """
        Удаляет аннотацию по пути к файлу
        
        Args:
            file_path: Относительный путь к файлу со свечами
        
        Returns:
            True если удалено, False если не найдено
        """
        mask = self.annotations['file'] == file_path
        if mask.any():
            self.annotations = self.annotations[~mask].copy()
            self.save_annotations()
            print(f"✅ Аннотация удалена: {file_path}")
            return True
        else:
            print(f"❌ Аннотация не найдена: {file_path}")
            return False
    
    def annotate_from_scanner(self, df, ticker, timeframe, pattern_info, label=None):
        """
        Автоматическая аннотация из результатов сканера
        
        Args:
            df: DataFrame со свечами
            ticker: Тикер
            timeframe: Таймфрейм
            pattern_info: Результат сканера (dict с информацией о паттерне)
            label: Метка (если None, определяется автоматически)
        """
        # Определяем метку автоматически
        if label is None:
            if 'pattern' in pattern_info:
                if 'BEARISH' in pattern_info['pattern']:
                    label = 2  # Медвежий
                elif 'FLAG' in pattern_info['pattern']:
                    label = 1  # Бычий
                else:
                    label = 0  # Нет паттерна
            else:
                label = 0
        
        # Сохраняем свечи
        candles_file = self.save_candles(df, ticker, timeframe, pattern_info)
        
        # Добавляем аннотацию
        pattern_type = pattern_info.get('pattern', 'UNKNOWN')
        notes = f"Quality score: {pattern_info.get('quality_score', 'N/A')}"
        
        self.annotate_pattern(
            candles_file, label, ticker, timeframe, 
            pattern_type=pattern_type, notes=notes
        )
        
        return candles_file
    
    def annotate_false_positive(self, df, ticker, timeframe, scanner_result):
        """
        Помечает результат сканера как ложное срабатывание (нет паттерна)
        
        Args:
            df: DataFrame со свечами
            ticker: Тикер
            timeframe: Таймфрейм
            scanner_result: Результат сканера (который оказался ложным)
        """
        candles_file = self.save_candles(df, ticker, timeframe, scanner_result)
        
        self.annotate_pattern(
            candles_file, label=0, ticker=ticker, timeframe=timeframe,
            pattern_type='FALSE_POSITIVE',
            notes=f"Помечено как ложное срабатывание. Scanner result: {scanner_result.get('pattern', 'N/A')}"
        )
        
        return candles_file
    
    def save_annotations(self):
        """Сохраняет аннотации в CSV"""
        self.annotations.to_csv(self.annotations_file, index=False)
        print(f"💾 Аннотации сохранены: {self.annotations_file}")
        print(f"   Всего аннотаций: {len(self.annotations)}")
    
    def get_statistics(self):
        """Возвращает статистику по аннотациям"""
        if len(self.annotations) == 0:
            return {
                'total': 0,
                'by_label': {},
                'by_timeframe': {},
                'by_ticker': {}
            }
        
        stats = {
            'total': len(self.annotations),
            'by_label': self.annotations['label'].value_counts().to_dict(),
            'by_timeframe': self.annotations['timeframe'].value_counts().to_dict(),
            'by_ticker': self.annotations['ticker'].value_counts().to_dict()
        }
        
        return stats
    
    def print_statistics(self):
        """Выводит статистику"""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("СТАТИСТИКА АННОТАЦИЙ")
        print("="*60)
        print(f"Всего аннотаций: {stats['total']}")
        print(f"\nПо меткам:")
        for label, count in stats['by_label'].items():
            label_name = {0: 'Нет паттерна', 1: 'Бычий', 2: 'Медвежий'}.get(label, f'Unknown({label})')
            print(f"  {label_name}: {count}")
        print(f"\nПо таймфреймам:")
        for tf, count in stats['by_timeframe'].items():
            print(f"  {tf}: {count}")
        print("="*60)


if __name__ == "__main__":
    # Тестирование аннотатора
    print("Тестирование аннотатора...")
    
    annotator = PatternAnnotator()
    
    # Пример статистики
    annotator.print_statistics()


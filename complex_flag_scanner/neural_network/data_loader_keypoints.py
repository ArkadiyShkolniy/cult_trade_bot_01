"""
Загрузчик данных для обучения нейронной сети с keypoint detection
Преобразует данные свечей в изображения и загружает координаты ключевых точек
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')  # Неинтерактивный бэкенд
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
from PIL import Image


class FlagPatternKeypointDataset(Dataset):
    """
    Датасет для обучения сети на паттернах "Флаг" с детекцией ключевых точек
    """
    
    def __init__(self, data_dir, transform=None, image_size=(224, 224), window=100):
        """
        Args:
            data_dir: Директория с данными (CSV файлы и метки)
            transform: Трансформации для аугментации
            image_size: Размер выходного изображения (height, width)
            window: Количество свечей для отображения (если данных больше)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.image_size = image_size
        self.window = window
        
        # Загружаем список файлов с метками
        self.annotations_file = os.path.join(data_dir, 'annotations.csv')
        if os.path.exists(self.annotations_file):
            self.annotations = pd.read_csv(self.annotations_file)
            # Фильтруем только те, у которых есть координаты точек
            # (для keypoint detection нужны только примеры с паттернами)
            self.annotations = self.annotations[
                (self.annotations['t0_idx'].notna()) &
                (self.annotations['t1_idx'].notna()) &
                (self.annotations['t2_idx'].notna()) &
                (self.annotations['t3_idx'].notna()) &
                (self.annotations['t4_idx'].notna()) &
                (self.annotations['label'].isin([1, 2]))  # Только бычий или медвежий
            ].copy()
        else:
            self.annotations = pd.DataFrame()
            print("⚠️ Файл аннотаций не найден, создан пустой датасет")
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """Получает один элемент датасета"""
        row = self.annotations.iloc[idx]
        file_path = os.path.join(self.data_dir, row['file'])
        
        # Загружаем данные свечей
        df = pd.read_csv(file_path)
        
        # Преобразуем в изображение
        image, normalization_params = self._candles_to_image(df)
        
        # Загружаем координаты точек
        keypoints = self._load_keypoints(row, df, normalization_params)
        
        # Применяем трансформации
        if self.transform:
            image = self.transform(image)
        
        # Метка класса (1=бычий, 2=медвежий)
        label = int(row['label'])
        
        return image, label, keypoints
    
    def _load_keypoints(self, row, df, normalization_params):
        """
        Загружает координаты ключевых точек и преобразует в нормализованные координаты изображения
        
        Args:
            row: Строка из annotations.csv
            df: DataFrame со свечами
            normalization_params: Параметры нормализации (price_min, price_max, window_start)
        
        Returns:
            Тензор координат [5, 2] (x, y для каждой точки, нормализованные 0-1)
        """
        price_min = normalization_params['price_min']
        price_max = normalization_params['price_max']
        price_range = price_max - price_min
        window_start = normalization_params.get('window_start', 0)
        window_size = normalization_params.get('window_size', self.window)  # Используем реальную длину окна
        
        keypoints = []
        for point_name in ['T0', 'T1', 'T2', 'T3', 'T4']:
            point_lower = point_name.lower()
            idx_col = f'{point_lower}_idx'
            price_col = f'{point_lower}_price'
            
            if pd.notna(row[idx_col]) and pd.notna(row[price_col]):
                # Индекс свечи
                candle_idx = int(row[idx_col])
                
                # Цена точки
                point_price = float(row[price_col])
                
                # Нормализуем координаты
                # X: позиция свечи в окне (0-1)
                # ВАЖНО: используем window_size (реальная длина окна), а не self.window!
                # Если точка вне окна, она будет обрезана до 0 или 1, но это лучше чем неправильная нормализация
                if window_size > 1:
                    x_norm = (candle_idx - window_start) / (window_size - 1)  # Нормализуем к [0, 1] где 1 = последняя свеча
                else:
                    x_norm = 0.5  # Если окно из 1 свечи, ставим центр
                x_norm = max(0.0, min(1.0, x_norm))  # Ограничиваем 0-1
                
                # Y: нормализованная цена (0-1)
                if price_range > 0:
                    y_norm = (point_price - price_min) / price_range
                else:
                    y_norm = 0.5
                y_norm = max(0.0, min(1.0, y_norm))  # Ограничиваем 0-1
                
                keypoints.append([x_norm, y_norm])
            else:
                # Если координаты отсутствуют, используем центр
                keypoints.append([0.5, 0.5])
        
        return torch.tensor(keypoints, dtype=torch.float32)
    
    def _candles_to_image(self, df, window=None):
        """
        Преобразует свечи в изображение графика
        
        Args:
            df: DataFrame со свечами (columns: time, open, high, low, close, volume)
            window: Количество свечей для отображения (если данных больше)
        
        Returns:
            tuple: (img_tensor, normalization_params)
                img_tensor: Тензор изображения (3, H, W)
                normalization_params: dict с параметрами нормализации
        """
        if window is None:
            window = self.window
        
        # Берем последние window свечей
        if len(df) > window:
            df_plot = df.tail(window).copy()
            window_start = len(df) - window
        else:
            df_plot = df.copy()
            window_start = 0
        
        # Нормализуем цены (приводим к диапазону 0-1)
        price_min = df_plot[['low']].min().min()
        price_max = df_plot[['high']].max().max()
        price_range = price_max - price_min
        
        if price_range == 0:
            price_range = 1
        
        # Нормализуем данные
        df_normalized = df_plot.copy()
        for col in ['open', 'high', 'low', 'close']:
            df_normalized[col] = (df_plot[col] - price_min) / price_range
        
        # Нормализуем объем
        volume_max = df_plot['volume'].max()
        if volume_max > 0:
            df_normalized['volume'] = df_plot['volume'] / volume_max
        
        # Сохраняем параметры нормализации
        normalization_params = {
            'price_min': price_min,
            'price_max': price_max,
            'window_start': window_start,
            'window_size': len(df_plot)  # Реальная длина окна для правильной нормализации
        }
        
        # Создаем фигуру matplotlib
        fig = plt.figure(figsize=(10, 6), dpi=22.4)  # 224x224 пикселя при dpi=22.4
        fig.patch.set_facecolor('black')
        ax = fig.add_subplot(111)
        ax.set_facecolor('black')
        ax.axis('off')
        
        # Рисуем свечи
        for i, row in df_normalized.iterrows():
            idx = df_normalized.index.get_loc(i)
            open_price = row['open']
            close_price = row['close']
            high_price = row['high']
            low_price = row['low']
            
            # Цвет свечи
            color = 'green' if close_price >= open_price else 'red'
            
            # Тело свечи
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            rect = plt.Rectangle((idx - 0.3, body_bottom), 0.6, body_height, 
                               facecolor=color, edgecolor=color, linewidth=0.5)
            ax.add_patch(rect)
            
            # Тени
            ax.plot([idx, idx], [low_price, high_price], color=color, linewidth=1)
        
        # Убираем отступы
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Конвертируем в PIL Image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        img_array = np.asarray(buf)
        img_pil = Image.fromarray(img_array).convert('RGB')
        
        # Закрываем фигуру
        plt.close(fig)
        
        # Изменяем размер
        img_pil = img_pil.resize(self.image_size, Image.LANCZOS)
        
        # Конвертируем в numpy array и нормализуем
        img_array = np.array(img_pil).astype(np.float32) / 255.0
        
        # Преобразуем в формат (C, H, W)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        
        return img_tensor, normalization_params


def create_keypoint_data_loader(data_dir, batch_size=32, shuffle=True, transform=None, 
                                image_size=(224, 224), train_split=0.8, window=100):
    """
    Создает DataLoader для обучения с keypoint detection
    
    Args:
        data_dir: Директория с данными
        batch_size: Размер батча
        shuffle: Перемешивать ли данные
        transform: Трансформации
        image_size: Размер изображения
        train_split: Доля данных для обучения (остальное для валидации)
        window: Количество свечей в окне
    
    Returns:
        train_loader, val_loader
    """
    dataset = FlagPatternKeypointDataset(data_dir, transform=transform, 
                                        image_size=image_size, window=window)
    
    # Разделение на train/val
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    if train_size > 0 and val_size > 0:
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    else:
        # Если данных мало, используем весь датасет для обучения
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return train_loader, None


if __name__ == "__main__":
    # Тестирование загрузчика
    print("Тестирование загрузчика данных для keypoint detection...")
    
    data_dir = "neural_network/data"
    if os.path.exists(data_dir):
        dataset = FlagPatternKeypointDataset(data_dir)
        print(f"Размер датасета: {len(dataset)}")
        
        if len(dataset) > 0:
            image, label, keypoints = dataset[0]
            print(f"Размер изображения: {image.shape}")
            print(f"Метка: {label}")
            print(f"Ключевые точки: {keypoints.shape}")
            print(f"Координаты точек:\n{keypoints}")
    else:
        print(f"Директория {data_dir} не найдена")


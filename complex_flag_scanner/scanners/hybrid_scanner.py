"""
Объединенный сканер, использующий математический анализ и нейронную сеть
"""
import os
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from .combined_scanner import ComplexFlagScanner
try:
    from neural_network.model_keypoints import create_keypoint_model
    from neural_network.data_loader_keypoints import FlagPatternKeypointDataset
    NN_AVAILABLE = True
except ImportError:
    NN_AVAILABLE = False
    print("⚠️  Нейронная сеть недоступна (модули не найдены)")


class HybridFlagScanner(ComplexFlagScanner):
    """
    Объединенный сканер, использующий:
    - Математический анализ для поиска паттернов
    - Нейронную сеть для оценки качества найденных паттернов
    """
    
    def __init__(self, token: str, model_path: str = None, device: str = 'cpu', 
                 nn_window: int = 100, nn_min_confidence: float = 0.5, use_nn: bool = True):
        """
        Args:
            token: Tinkoff Invest API токен
            model_path: Путь к обученной модели (если None, используется default)
            device: Устройство для вычислений ('cpu' или 'cuda')
            nn_window: Размер окна для нейронной сети
            nn_min_confidence: Минимальная уверенность нейронной сети
            use_nn: Использовать ли нейронную сеть (если False, работает как обычный математический сканер)
        """
        super().__init__(token)
        self.use_nn = use_nn
        self.nn_window = nn_window
        self.nn_min_confidence = nn_min_confidence
        self.device = torch.device(device)
        
        if self.use_nn:
            if not NN_AVAILABLE:
                print("⚠️  Нейронная сеть недоступна, используется только математический анализ")
                self.use_nn = False
                self.nn_model = None
            else:
                # Загружаем модель нейронной сети
                if model_path is None:
                    model_path = 'neural_network/models/keypoint_model_best.pth'
                    # Проверяем относительный путь
                    if not os.path.exists(model_path):
                        model_path = str(Path(__file__).parent.parent / model_path)
                
                if os.path.exists(model_path):
                    print(f"🏗️  Загрузка модели: {model_path}")
                    self.nn_model = create_keypoint_model(
                        num_classes=3,
                        num_keypoints=5,
                        image_height=224,
                        image_width=224,
                        pretrained_path=model_path
                    )
                    self.nn_model.to(self.device)
                    self.nn_model.eval()
                    print(f"   ✅ Модель загружена (устройство: {self.device})")
                else:
                    print(f"⚠️  Модель не найдена: {model_path}, нейронная сеть отключена")
                    self.use_nn = False
                    self.nn_model = None
        else:
            self.nn_model = None
    
    def _evaluate_pattern_with_nn(self, df, pattern_info, window_start: int = 0):
        """
        Оценивает паттерн найденный математическим сканером с помощью нейронной сети
        
        Args:
            df: DataFrame со свечами
            pattern_info: Информация о паттерне от математического сканера
            window_start: Начальный индекс окна для анализа
            
        Returns:
            dict: {
                'nn_confidence': float,  # Уверенность нейронной сети (0-1)
                'nn_class': int,         # Класс от нейронной сети (1=бычий, 2=медвежий)
                'nn_match': bool         # Совпадает ли класс с математическим сканером
            }
        """
        if not self.use_nn or self.nn_model is None:
            return {
                'nn_confidence': 0.0,
                'nn_class': 0,
                'nn_match': False
            }
        
        try:
            # Определяем индексы точек паттерна
            t0_idx = pattern_info.get('t0', {}).get('idx', 0)
            t4_idx = pattern_info.get('t4', {}).get('idx', 0)
            
            # Определяем окно для анализа (центрируем на паттерне)
            pattern_center = (t0_idx + t4_idx) // 2
            window_half = self.nn_window // 2
            
            start_idx = max(0, pattern_center - window_half)
            end_idx = min(len(df), start_idx + self.nn_window)
            
            # Если окно выходит за пределы, сдвигаем
            if end_idx - start_idx < self.nn_window:
                end_idx = min(len(df), start_idx + self.nn_window)
                start_idx = max(0, end_idx - self.nn_window)
            
            df_window = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
            
            if len(df_window) < 50:  # Минимум свечей для анализа
                return {
                    'nn_confidence': 0.0,
                    'nn_class': 0,
                    'nn_match': False
                }
            
            # Преобразуем в изображение
            dataset = FlagPatternKeypointDataset("", image_size=(224, 224), window=len(df_window))
            image_tensor, _ = dataset._candles_to_image(df_window, window=len(df_window))
            
            # Предсказание нейронной сети
            with torch.no_grad():
                image_batch = image_tensor.unsqueeze(0).to(self.device)
                class_logits, _ = self.nn_model(image_batch)
                probabilities = F.softmax(class_logits, dim=1)
                predicted_class = torch.argmax(class_logits, dim=1).item()
                pred_prob = probabilities[0][predicted_class].item()
            
            # Определяем класс паттерна от математического сканера
            pattern_name = pattern_info.get('pattern', '')
            math_class = 1 if 'BEARISH' not in pattern_name and 'FLAG' in pattern_name else 2 if 'BEARISH' in pattern_name else 0
            
            # Проверяем совпадение классов
            nn_match = (predicted_class == math_class) and (predicted_class > 0)
            
            return {
                'nn_confidence': pred_prob,
                'nn_class': predicted_class,
                'nn_match': nn_match
            }
        except Exception as e:
            print(f"   ⚠️ Ошибка оценки паттерна нейронной сетью: {e}")
            return {
                'nn_confidence': 0.0,
                'nn_class': 0,
                'nn_match': False
            }
    
    def analyze(self, df, debug=False, timeframe='1h', 
                filter_by_nn: bool = True, min_nn_confidence: float = None):
        """
        Анализирует паттерны используя математический сканер и нейронную сеть
        
        Args:
            df: DataFrame со свечами
            debug: Режим отладки
            timeframe: Таймфрейм
            filter_by_nn: Фильтровать ли результаты по нейронной сети
            min_nn_confidence: Минимальная уверенность нейронной сети (если None, используется self.nn_min_confidence)
        
        Returns:
            list: Список паттернов с дополнительной информацией от нейронной сети
        """
        # Используем математический сканер для поиска паттернов
        math_patterns = super().analyze(df, debug=debug, timeframe=timeframe)
        
        if not math_patterns:
            return []
        
        if not self.use_nn:
            # Если нейронная сеть не используется, возвращаем результаты математического сканера
            for pattern in math_patterns:
                pattern['nn_confidence'] = 0.0
                pattern['nn_class'] = 0
                pattern['nn_match'] = False
            return math_patterns
        
        # Оцениваем каждый паттерн нейронной сетью
        min_confidence = min_nn_confidence if min_nn_confidence is not None else self.nn_min_confidence
        evaluated_patterns = []
        
        for pattern in math_patterns:
            nn_result = self._evaluate_pattern_with_nn(df, pattern)
            
            # Добавляем информацию от нейронной сети
            pattern['nn_confidence'] = nn_result['nn_confidence']
            pattern['nn_class'] = nn_result['nn_class']
            pattern['nn_match'] = nn_result['nn_match']
            
            # Фильтруем по уверенности и совпадению классов
            if filter_by_nn:
                if nn_result['nn_confidence'] >= min_confidence and nn_result['nn_match']:
                    evaluated_patterns.append(pattern)
            else:
                # Не фильтруем, просто добавляем информацию
                evaluated_patterns.append(pattern)
        
        return evaluated_patterns
    
    def analyze_with_nn_only(self, df, window: int = None, step: int = None, min_confidence: float = None):
        """
        Анализирует паттерны используя только нейронную сеть (sliding window)
        
        Args:
            df: DataFrame со свечами
            window: Размер окна (если None, используется self.nn_window)
            step: Шаг скользящего окна (если None, используется window // 10)
            min_confidence: Минимальная уверенность (если None, используется self.nn_min_confidence)
        
        Returns:
            list: Список паттернов найденных нейронной сетью
        """
        if not self.use_nn or self.nn_model is None:
            return []
        
        if not NN_AVAILABLE:
            return []
        
        from neural_network.predict_keypoints import predict_with_sliding_window
        
        window_size = window or self.nn_window
        step_size = step or max(10, window_size // 10)
        min_conf = min_confidence or self.nn_min_confidence
        
        predictions = predict_with_sliding_window(
            df, 
            self.nn_model, 
            window=window_size, 
            step=step_size, 
            device=self.device, 
            min_confidence=min_conf
        )
        
        # Преобразуем формат предсказаний в формат математического сканера
        patterns = []
        for pred in predictions:
            points = pred['points']
            if len(points) == 5:
                pattern = {
                    'pattern': 'BULLISH_FLAG' if pred['class'] == 1 else 'BEARISH_FLAG',
                    't0': {'idx': points[0]['idx'], 'price': points[0]['price']},
                    't1': {'idx': points[1]['idx'], 'price': points[1]['price']},
                    't2': {'idx': points[2]['idx'], 'price': points[2]['price']},
                    't3': {'idx': points[3]['idx'], 'price': points[3]['price']},
                    't4': {'idx': points[4]['idx'], 'price': points[4]['price']},
                    'nn_confidence': pred['probability'],
                    'nn_class': pred['class'],
                    'nn_match': True,  # Всегда True для NN-only
                    'source': 'neural_network'
                }
                patterns.append(pattern)
        
        return patterns


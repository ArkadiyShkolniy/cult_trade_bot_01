"""
Тренировщик для модели с keypoint detection
Комбинированный loss: классификация + регрессия координат
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np


class KeypointModelTrainer:
    """
    Класс для обучения модели с keypoint detection
    """
    
    def __init__(self, model, device='cpu', classification_weight=1.0, keypoint_weight=1.0, 
                 order_penalty_weight=0.5, geometry_penalty_weight=0.5, tolerance_normalized=0.003):
        """
        Args:
            model: Модель для обучения
            device: Устройство для вычислений
            classification_weight: Вес для loss классификации
            keypoint_weight: Вес для loss регрессии ключевых точек
            order_penalty_weight: Вес для penalty за нарушение порядка точек (0.0 = отключено)
            geometry_penalty_weight: Вес для penalty за нарушение геометрических правил (0.0 = отключено)
            tolerance_normalized: Погрешность в нормализованных координатах (0.003 = 0.3% для 1h, соответствует реальной погрешности)
        """
        self.model = model
        self.device = device
        self.classification_weight = classification_weight
        self.keypoint_weight = keypoint_weight
        self.order_penalty_weight = order_penalty_weight
        self.geometry_penalty_weight = geometry_penalty_weight
        self.tolerance_normalized = tolerance_normalized  # Погрешность для нормализованных координат (0-1)
        
        # Loss функции
        self.classification_criterion = nn.CrossEntropyLoss()
        self.keypoint_criterion = nn.MSELoss()
        
        # Метрики
        self.train_losses = []
        self.val_losses = []
        self.train_classification_losses = []
        self.train_keypoint_losses = []
        self.train_order_penalties = []  # Метрики для order penalty
        self.train_geometry_penalties = []  # Метрики для geometry penalty
        self.val_classification_losses = []
        self.val_keypoint_losses = []
        self.val_order_penalties = []  # Метрики для order penalty
        self.val_geometry_penalties = []  # Метрики для geometry penalty
        self.train_accuracies = []
        self.val_accuracies = []
        
    def _compute_order_penalty(self, pred_keypoints, mask):
        """
        Вычисляет penalty за нарушение порядка точек T0 < T1 < T2 < T3 < T4
        
        Args:
            pred_keypoints: [batch, 5, 2] - предсказанные координаты (x, y)
            mask: [batch, 1, 1] - маска для примеров с паттерном (label > 0)
        
        Returns:
            penalty: скаляр - средний penalty за нарушение порядка
        """
        if self.order_penalty_weight == 0.0:
            return torch.tensor(0.0, device=pred_keypoints.device)
        
        # Применяем маску (только для примеров с паттерном)
        masked_keypoints = pred_keypoints * mask
        
        # Берем X координаты (индекс 0)
        x_coords = masked_keypoints[:, :, 0]  # [batch, 5]
        
        penalty = torch.tensor(0.0, device=pred_keypoints.device)
        
        # Проверяем порядок: T0 < T1 < T2 < T3 < T4
        # Добавляем минимальный отступ (margin) между точками, чтобы они не слипались
        # В окне 100 свечей отступ 0.02 = 2 свечи
        min_dist = 0.02 
        
        for i in range(4):  # Проверяем 4 пары: T0<T1, T1<T2, T2<T3, T3<T4
            # T[i+1].x должно быть > T[i].x + min_dist
            # Violation если T[i+1].x - T[i].x < min_dist
            # Т.е. T[i].x - T[i+1].x > -min_dist
            # Штраф = max(0, T[i].x + min_dist - T[i+1].x)
            
            diff = x_coords[:, i+1] - x_coords[:, i]
            violation = torch.clamp(min_dist - diff, min=0.0)
            penalty += violation.sum() * 5.0 # Увеличиваем вес штрафа за порядок
        
        # Нормализуем по количеству активных примеров (с паттерном)
        active_count = mask.sum()
        if active_count > 0:
            penalty = penalty / active_count
        else:
            penalty = torch.tensor(0.0, device=pred_keypoints.device)
        
        return penalty
    
    def _compute_geometry_penalty(self, pred_keypoints, labels, mask):
        """
        Вычисляет penalty за нарушение геометрических правил паттерна флаг.
        Учитывает взаимосвязи всех 5 точек паттерна.
        
        Args:
            pred_keypoints: [batch, 5, 2] - предсказанные координаты (x, y)
            labels: [batch] - метки классов (0=нет паттерна, 1=бычий, 2=медвежий)
            mask: [batch, 1, 1] - маска для примеров с паттерном
        
        Returns:
            penalty: скаляр - средний penalty за геометрические нарушения
        """
        if self.geometry_penalty_weight == 0.0:
            return torch.tensor(0.0, device=pred_keypoints.device)
        
        # Применяем маску (только для примеров с паттерном)
        masked_keypoints = pred_keypoints * mask
        
        # Берем координаты
        x_coords = masked_keypoints[:, :, 0]  # [batch, 5] - X координаты (время/индекс)
        y_coords = masked_keypoints[:, :, 1]  # [batch, 5] - Y координаты (цена)
        
        penalty = torch.tensor(0.0, device=pred_keypoints.device)
        active_count = 0
        
        for i in range(len(pred_keypoints)):
            # Пропускаем примеры без паттерна (label == 0)
            if labels[i].item() == 0:
                continue
            
            active_count += 1
            label = labels[i].item()
            
            # Извлекаем координаты точек для этого паттерна
            t0_x, t0_y = x_coords[i, 0], y_coords[i, 0]
            t1_x, t1_y = x_coords[i, 1], y_coords[i, 1]
            t2_x, t2_y = x_coords[i, 2], y_coords[i, 2]
            t3_x, t3_y = x_coords[i, 3], y_coords[i, 3]
            t4_x, t4_y = x_coords[i, 4], y_coords[i, 4]
            
            # Высота флагштока
            pole_height = torch.abs(t1_y - t0_y)
            
            if label == 1:  # Бычий флаг (LONG)
                # Правило 1: T0 - нет ограничений (пропускаем)
                
                # Правило 2: T1 - нет ограничений (пропускаем)
                
                # Правило 3: T2 - не ниже фиба 0.62 хода T0-T1
                # Коррекция фибоначчи 0.62 от вершины T1 вниз: T2 >= T1 - 0.62 * (T1 - T0)
                min_t2 = t1_y - 0.62 * (t1_y - t0_y)
                tolerance_t2 = min_t2 * self.tolerance_normalized
                if t2_y < min_t2 - tolerance_t2:
                    violation = torch.clamp(min_t2 - tolerance_t2 - t2_y, min=0.0)
                    penalty += violation * 2.0  # Сильный штраф
                
                # Правило 4: T3 должна находиться в диапазоне фиба 0.5 хода T1-T2, но не выше T1
                # T2 + 0.5 * (T1 - T2) <= T3 <= T1
                # Правило: T3 должна быть в диапазоне между уровнем фиба 0.5 и T1
                # Ход T1-T2 = T1 - T2 (для бычьего флага T1 > T2)
                move_12 = t1_y - t2_y  # Ход T1-T2
                fib_50_level = t2_y + 0.5 * move_12  # Уровень фиба 0.5
                tolerance_t3_min = fib_50_level * self.tolerance_normalized
                tolerance_t1 = t1_y * self.tolerance_normalized
                
                # T3 должна быть >= фиба 0.5 И <= T1
                if t3_y < fib_50_level - tolerance_t3_min:
                    violation = torch.clamp(fib_50_level - tolerance_t3_min - t3_y, min=0.0)
                    penalty += violation * 1.5
                if t3_y > t1_y + tolerance_t1:
                    violation = torch.clamp(t3_y - t1_y - tolerance_t1, min=0.0)
                    penalty += violation * 1.5
                
                # Правило 5: T4 - минимальный ход фиба 0.5 хода T2-T3, но не ниже фиба 0.62 хода T0-T1
                # Ход T2-T3 = T3 - T2 (для бычьего флага T3 > T2)
                # T4 должна быть ниже T3 (откат), но не ниже min_t2 (фиба 0.62 от T0-T1)
                move_23 = t3_y - t2_y  # Ход T2-T3
                # Минимальный откат от T3: T4 <= T3 - 0.5 * move_23
                max_t4_from_t3 = t3_y - 0.5 * move_23  # Максимальный откат от T3 (50% от хода T2-T3)
                min_t4_from_pole = t1_y - 0.62 * (t1_y - t0_y)  # Коррекция фибоначчи 0.62 от T1 вниз
                tolerance_t4_max = max_t4_from_t3 * self.tolerance_normalized
                tolerance_t4_min = min_t4_from_pole * self.tolerance_normalized
                
                # T4 должна быть <= max_t4_from_t3 (не больше отката) И >= min_t4_from_pole (не ниже фиба 0.62)
                if t4_y > max_t4_from_t3 + tolerance_t4_max:
                    violation = torch.clamp(t4_y - max_t4_from_t3 - tolerance_t4_max, min=0.0)
                    penalty += violation * 1.5
                if t4_y < min_t4_from_pole - tolerance_t4_min:
                    violation = torch.clamp(min_t4_from_pole - tolerance_t4_min - t4_y, min=0.0)
                    penalty += violation * 2.0  # Сильный штраф
                
                # Правило 6: Линии T1-T3 и T2-T4 могут быть параллельны или сходиться, но не могут расходиться
                # Вычисляем углы наклона линий (в нормализованных координатах)
                if torch.abs(t3_x - t1_x) > 1e-6 and torch.abs(t4_x - t2_x) > 1e-6:
                    slope_13 = (t3_y - t1_y) / (t3_x - t1_x)  # Угол линии T1-T3
                    slope_24 = (t4_y - t2_y) / (t4_x - t2_x)  # Угол линии T2-T4
                    
                    # Для бычьего: линии должны быть параллельны или сходиться
                    # Если slope_24 < slope_13 * 0.98, значит линии расходятся (порог 2% для допуска)
                    # Для бычьего флага обе линии направлены вниз (slope < 0), и для схождения slope_24 должен быть менее отрицательным
                    if slope_13 < 0 and slope_24 < slope_13 * 0.98:
                        divergence = slope_13 * 0.98 - slope_24
                        penalty += torch.clamp(divergence, min=0.0) * 1.0
            
            elif label == 2:  # Медвежий флаг (SHORT)
                # Правило 1: T0 - нет ограничений (пропускаем)
                
                # Правило 2: T1 - нет ограничений (пропускаем)
                
                # Правило 3: T2 - не выше фиба 0.62 хода T0-T1
                # Коррекция фибоначчи 0.62 от низа T1 вверх: T2 <= T1 + 0.62 * (T0 - T1)
                max_t2 = t1_y + 0.62 * (t0_y - t1_y)
                tolerance_t2 = max_t2 * self.tolerance_normalized
                if t2_y > max_t2 + tolerance_t2:
                    violation = torch.clamp(t2_y - max_t2 - tolerance_t2, min=0.0)
                    penalty += violation * 2.0  # Сильный штраф
                
                # Правило 4: T3 должна находиться в диапазоне фиба 0.5 хода T1-T2, но не ниже T1
                # T1 <= T3 <= T1 + 0.5 * (T2 - T1) = фиба 0.5
                # Правило: T3 должна быть в диапазоне между T1 и уровнем фиба 0.5
                # Ход T1-T2 = T2 - T1 (для медвежьего флага T2 > T1)
                move_12 = t2_y - t1_y  # Ход T1-T2
                fib_50_level = t1_y + 0.5 * move_12  # Уровень фиба 0.5
                tolerance_t1 = t1_y * self.tolerance_normalized
                tolerance_t3_max = fib_50_level * self.tolerance_normalized
                
                # T3 должна быть >= T1 (обязательно)
                if t3_y < t1_y - tolerance_t1:
                    violation = torch.clamp(t1_y - tolerance_t1 - t3_y, min=0.0)
                    penalty += violation * 1.5
                
                # Проверяем, что T3 <= фиба 0.5 (верхняя граница диапазона)
                if t3_y > fib_50_level + tolerance_t3_max:
                    violation = torch.clamp(t3_y - fib_50_level - tolerance_t3_max, min=0.0)
                    penalty += violation * 1.5
                
                # Правило 5: T4 - минимальный ход фиба 0.5 хода T2-T3, но не выше фиба 0.62 хода T0-T1
                # Ход T2-T3 = T2 - T3 (для медвежьего флага T2 > T3)
                # T4 должна быть выше T3 (откат вверх), но не выше max_t2 (фиба 0.62 от T0-T1)
                move_23 = t2_y - t3_y  # Ход T2-T3
                # Минимальный откат от T3: T4 >= T3 + 0.5 * move_23
                min_t4_from_t3 = t3_y + 0.5 * move_23  # Минимальный откат от T3 (50% от хода T2-T3)
                max_t4_from_pole = t1_y + 0.62 * (t0_y - t1_y)  # Коррекция фибоначчи 0.62 от T1 вверх
                tolerance_t4_min = min_t4_from_t3 * self.tolerance_normalized
                tolerance_t4_max = max_t4_from_pole * self.tolerance_normalized
                
                # T4 должна быть >= min_t4_from_t3 (не меньше отката) И <= max_t4_from_pole (не выше фиба 0.62)
                if t4_y < min_t4_from_t3 - tolerance_t4_min:
                    violation = torch.clamp(min_t4_from_t3 - tolerance_t4_min - t4_y, min=0.0)
                    penalty += violation * 1.5
                if t4_y > max_t4_from_pole + tolerance_t4_max:
                    violation = torch.clamp(t4_y - max_t4_from_pole - tolerance_t4_max, min=0.0)
                    penalty += violation * 2.0  # Сильный штраф
                
                # Правило 6: Линии T1-T3 и T2-T4 могут быть параллельны или сходиться, но не могут расходиться
                # Вычисляем углы наклона линий (в нормализованных координатах)
                if torch.abs(t3_x - t1_x) > 1e-6 and torch.abs(t4_x - t2_x) > 1e-6:
                    slope_13 = (t3_y - t1_y) / (t3_x - t1_x)  # Угол линии T1-T3
                    slope_24 = (t4_y - t2_y) / (t4_x - t2_x)  # Угол линии T2-T4
                    
                    # Для медвежьего: линии должны быть параллельны или сходиться
                    # Если slope_24 > slope_13 * 1.02, значит линии расходятся (порог 2% для допуска)
                    # Для медвежьего флага обе линии направлены вверх (slope > 0), и для схождения slope_24 должен быть менее положительным
                    if slope_13 > 0 and slope_24 > slope_13 * 1.02:
                        divergence = slope_24 - slope_13 * 1.02
                        penalty += torch.clamp(divergence, min=0.0) * 1.0
            
            # Общее правило: минимальная высота флагштока (для обоих типов)
            # Флагшток должен быть достаточно длинным (нормализованные координаты)
            # Увеличиваем требование до 0.15 (15% от окна) и вес штрафа
            min_pole_height = 0.15  
            if pole_height < min_pole_height:
                violation = min_pole_height - pole_height
                penalty += violation * 5.0 # Усиленный штраф за маленький флагшток
        
        # Нормализуем по количеству активных примеров
        if active_count > 0:
            penalty = penalty / active_count
        else:
            penalty = torch.tensor(0.0, device=pred_keypoints.device)
        
        return penalty
    
    def train_epoch(self, train_loader, optimizer):
        """Одна эпоха обучения"""
        self.model.train()
        total_loss = 0.0
        total_classification_loss = 0.0
        total_keypoint_loss = 0.0
        total_order_penalty = 0.0
        total_geometry_penalty = 0.0
        correct = 0
        total = 0
        
        for images, labels, keypoints in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            keypoints = keypoints.to(self.device)
            
            # Обнуляем градиенты
            optimizer.zero_grad()
            
            # Прямой проход
            class_logits, pred_keypoints = self.model(images)
            
            # Loss для классификации
            classification_loss = self.classification_criterion(class_logits, labels)
            
            # Loss для ключевых точек (только для примеров с паттерном: label 1 или 2)
            # Для label 0 (нет паттерна) ключевые точки не важны
            mask = (labels > 0).float().unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1]
            
            # MSE loss для координат
            mse_loss = self.keypoint_criterion(pred_keypoints * mask, keypoints * mask)
            
            # Penalty за нарушение порядка точек
            order_penalty = self._compute_order_penalty(pred_keypoints, mask)
            
            # Penalty за нарушение геометрических правил
            geometry_penalty = self._compute_geometry_penalty(pred_keypoints, labels, mask)
            
            # Комбинированный keypoint loss
            keypoint_loss = (mse_loss + 
                           self.order_penalty_weight * order_penalty +
                           self.geometry_penalty_weight * geometry_penalty)
            
            # Комбинированный loss
            loss = (self.classification_weight * classification_loss + 
                   self.keypoint_weight * keypoint_loss)
            
            # Обратный проход
            loss.backward()
            optimizer.step()
            
            # Статистика
            total_loss += loss.item()
            total_classification_loss += classification_loss.item()
            total_keypoint_loss += mse_loss.item()
            total_order_penalty += order_penalty.item()
            total_geometry_penalty += geometry_penalty.item()
            
            # Точность классификации
            _, predicted = torch.max(class_logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        avg_classification_loss = total_classification_loss / len(train_loader)
        avg_keypoint_loss = total_keypoint_loss / len(train_loader)
        avg_order_penalty = total_order_penalty / len(train_loader)
        avg_geometry_penalty = total_geometry_penalty / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, avg_classification_loss, avg_keypoint_loss, avg_order_penalty, avg_geometry_penalty, accuracy
    
    def validate(self, val_loader):
        """Валидация"""
        self.model.eval()
        total_loss = 0.0
        total_classification_loss = 0.0
        total_keypoint_loss = 0.0
        total_order_penalty = 0.0
        total_geometry_penalty = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels, keypoints in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                keypoints = keypoints.to(self.device)
                
                # Прямой проход
                class_logits, pred_keypoints = self.model(images)
                
                # Loss для классификации
                classification_loss = self.classification_criterion(class_logits, labels)
                
                # Loss для ключевых точек
                mask = (labels > 0).float().unsqueeze(-1).unsqueeze(-1)
                
                # MSE loss для координат
                mse_loss = self.keypoint_criterion(pred_keypoints * mask, keypoints * mask)
                
                # Penalty за нарушение порядка точек
                order_penalty = self._compute_order_penalty(pred_keypoints, mask)
                
                # Penalty за нарушение геометрических правил
                geometry_penalty = self._compute_geometry_penalty(pred_keypoints, labels, mask)
                
                # Penalty за нарушение геометрических правил
                geometry_penalty = self._compute_geometry_penalty(pred_keypoints, labels, mask)
                
                # Комбинированный keypoint loss
                keypoint_loss = (mse_loss + 
                               self.order_penalty_weight * order_penalty +
                               self.geometry_penalty_weight * geometry_penalty)
                
                # Комбинированный loss
                loss = (self.classification_weight * classification_loss + 
                       self.keypoint_weight * keypoint_loss)
                
                # Статистика
                total_loss += loss.item()
                total_classification_loss += classification_loss.item()
                total_keypoint_loss += mse_loss.item()
                total_order_penalty += order_penalty.item()
                total_geometry_penalty += geometry_penalty.item()
                
                # Точность классификации
                _, predicted = torch.max(class_logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        avg_classification_loss = total_classification_loss / len(val_loader)
        avg_keypoint_loss = total_keypoint_loss / len(val_loader)
        avg_order_penalty = total_order_penalty / len(val_loader)
        avg_geometry_penalty = total_geometry_penalty / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, avg_classification_loss, avg_keypoint_loss, avg_order_penalty, avg_geometry_penalty, accuracy
    
    def train(self, train_loader, val_loader, epochs, optimizer, scheduler=None, 
              save_dir='neural_network/models', save_prefix='keypoint_model'):
        """
        Полный цикл обучения
        
        Args:
            train_loader: DataLoader для обучения
            val_loader: DataLoader для валидации
            epochs: Количество эпох
            optimizer: Оптимизатор
            scheduler: Планировщик learning rate (опционально)
            save_dir: Директория для сохранения моделей
            save_prefix: Префикс имени файла
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Обучение
            train_loss, train_cls_loss, train_kp_loss, train_order_penalty, train_geometry_penalty, train_acc = self.train_epoch(train_loader, optimizer)
            
            # Валидация
            if val_loader is not None:
                val_loss, val_cls_loss, val_kp_loss, val_order_penalty, val_geometry_penalty, val_acc = self.validate(val_loader)
            else:
                val_loss, val_cls_loss, val_kp_loss, val_order_penalty, val_geometry_penalty, val_acc = train_loss, train_cls_loss, train_kp_loss, train_order_penalty, train_geometry_penalty, train_acc
            
            # Сохраняем метрики
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_classification_losses.append(train_cls_loss)
            self.train_keypoint_losses.append(train_kp_loss)
            self.train_order_penalties.append(train_order_penalty)
            self.train_geometry_penalties.append(train_geometry_penalty)
            self.val_classification_losses.append(val_cls_loss)
            self.val_keypoint_losses.append(val_kp_loss)
            self.val_order_penalties.append(val_order_penalty)
            self.val_geometry_penalties.append(val_geometry_penalty)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Обновляем learning rate
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Выводим статистику
            print(f"Epoch {epoch+1}/{epochs}:")
            penalty_info = ""
            if self.order_penalty_weight > 0:
                penalty_info += f", Order: {train_order_penalty:.4f}"
            if self.geometry_penalty_weight > 0:
                penalty_info += f", Geometry: {train_geometry_penalty:.4f}"
            print(f"  Train Loss: {train_loss:.4f} (Cls: {train_cls_loss:.4f}, KP: {train_kp_loss:.4f}{penalty_info}) | Acc: {train_acc:.2f}%")
            if val_loader is not None:
                val_penalty_info = ""
                if self.order_penalty_weight > 0:
                    val_penalty_info += f", Order: {val_order_penalty:.4f}"
                if self.geometry_penalty_weight > 0:
                    val_penalty_info += f", Geometry: {val_geometry_penalty:.4f}"
                print(f"  Val Loss:   {val_loss:.4f} (Cls: {val_cls_loss:.4f}, KP: {val_kp_loss:.4f}{val_penalty_info}) | Acc: {val_acc:.2f}%")
            
            # Сохраняем лучшую модель
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(save_dir, f'{save_prefix}_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                }, best_model_path)
                print(f"  ✅ Сохранена лучшая модель: {best_model_path}")
            
            # Сохраняем последнюю модель
            last_model_path = os.path.join(save_dir, f'{save_prefix}_last.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, last_model_path)
            
            print()


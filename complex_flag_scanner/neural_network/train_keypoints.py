#!/usr/bin/env python3
"""
Скрипт для обучения нейронной сети с keypoint detection на размеченных данных паттернов "Флаг"
"""

import os
import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Добавляем путь к корню проекта
sys.path.insert(0, str(Path(__file__).parent.parent))

from neural_network.model_keypoints import create_keypoint_model
from neural_network.data_loader_keypoints import create_keypoint_data_loader
from neural_network.trainer_keypoints import KeypointModelTrainer


def main():
    parser = argparse.ArgumentParser(description='Обучение нейронной сети с keypoint detection для распознавания паттернов "Флаг"')
    parser.add_argument('--data_dir', type=str, default='neural_network/data',
                        help='Директория с данными (default: neural_network/data)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Количество эпох (default: 50)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Размер батча (default: 8)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Скорость обучения (default: 0.001)')
    parser.add_argument('--save_dir', type=str, default='neural_network/models',
                        help='Директория для сохранения моделей (default: neural_network/models)')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Размер изображения (default: 224)')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Количество классов (default: 3: 0=нет паттерна, 1=бычий, 2=медвежий)')
    parser.add_argument('--num_keypoints', type=int, default=5,
                        help='Количество ключевых точек (default: 5: T0-T4)')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Доля данных для обучения (default: 0.8)')
    parser.add_argument('--classification_weight', type=float, default=1.0,
                        help='Вес для loss классификации (default: 1.0)')
    parser.add_argument('--keypoint_weight', type=float, default=1.0,
                        help='Вес для loss регрессии ключевых точек (default: 1.0)')
    parser.add_argument('--order_penalty_weight', type=float, default=0.5,
                        help='Вес для penalty за нарушение порядка точек T0<T1<T2<T3<T4 (default: 0.5, 0.0 = отключено)')
    parser.add_argument('--geometry_penalty_weight', type=float, default=0.5,
                        help='Вес для penalty за нарушение геометрических правил (default: 0.5, 0.0 = отключено)')
    parser.add_argument('--tolerance_normalized', type=float, default=0.003,
                        help='Погрешность в нормализованных координатах для геометрии (default: 0.003 = 0.3% для 1h)')
    parser.add_argument('--window', type=int, default=100,
                        help='Количество свечей в окне (default: 100)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Устройство для вычислений (cpu, cuda, или auto)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Путь к предобученной модели для дообучения')
    
    args = parser.parse_args()
    
    # Проверяем наличие данных
    data_dir = Path(args.data_dir)
    annotations_file = data_dir / 'annotations.csv'
    
    if not annotations_file.exists():
        print(f"❌ Файл аннотаций не найден: {annotations_file}")
        return
    
    # Создаем директорию для моделей
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Определяем устройство
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("=" * 60)
    print("🎓 ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ С KEYPOINT DETECTION")
    print("=" * 60)
    print()
    print(f"📁 Данные: {data_dir}")
    print(f"💾 Сохранение моделей: {save_dir}")
    print(f"⚙️  Параметры:")
    print(f"   • Эпох: {args.epochs}")
    print(f"   • Батч: {args.batch_size}")
    print(f"   • Learning rate: {args.learning_rate}")
    print(f"   • Размер изображения: {args.image_size}x{args.image_size}")
    print(f"   • Классов: {args.num_classes}")
    print(f"   • Ключевых точек: {args.num_keypoints}")
    print(f"   • Окно свечей: {args.window}")
    print(f"   • Вес классификации: {args.classification_weight}")
    print(f"   • Вес keypoints: {args.keypoint_weight}")
    print(f"   • Вес order penalty: {args.order_penalty_weight} {'(отключено)' if args.order_penalty_weight == 0.0 else ''}")
    print(f"   • Вес geometry penalty: {args.geometry_penalty_weight} {'(отключено)' if args.geometry_penalty_weight == 0.0 else ''}")
    print(f"   • Погрешность для геометрии (нормализованная): {args.tolerance_normalized} ({args.tolerance_normalized*100:.1f}%)")
    print(f"   • Train/Val split: {args.train_split:.1%}")
    print(f"   • Устройство: {device}")
    print()
    
    # Создаем датасет
    print("📊 Загрузка данных...")
    try:
        train_loader, val_loader = create_keypoint_data_loader(
            str(data_dir),
            batch_size=args.batch_size,
            shuffle=True,
            image_size=(args.image_size, args.image_size),
            train_split=args.train_split,
            window=args.window
        )
        
        print(f"   ✅ Загружено примеров для обучения: {len(train_loader.dataset)}")
        if val_loader is not None:
            print(f"   ✅ Загружено примеров для валидации: {len(val_loader.dataset)}")
        else:
            print(f"   ⚠️  Валидационный набор пуст (используется весь датасет для обучения)")
        
        # Проверяем распределение классов в train
        train_dataset = train_loader.dataset
        if hasattr(train_dataset, 'dataset'):
            # Если используется random_split
            actual_dataset = train_dataset.dataset
        else:
            actual_dataset = train_dataset
        
        labels = []
        for i in range(len(actual_dataset)):
            if i < len(train_loader.dataset):
                try:
                    _, label, _ = actual_dataset[i]
                    labels.append(label)
                except:
                    pass
        
        if labels:
            from collections import Counter
            label_counts = Counter(labels)
            print(f"   📈 Распределение классов в обучении:")
            for label, count in sorted(label_counts.items()):
                class_name = {0: 'нет паттерна', 1: 'бычий флаг', 2: 'медвежий флаг'}.get(label, f'класс {label}')
                print(f"      • {class_name}: {count} примеров")
        print()
        
    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Создаем модель
    print(f"🏗️  Создание модели...")
    model = create_keypoint_model(
        num_classes=args.num_classes,
        num_keypoints=args.num_keypoints,
        image_height=args.image_size,
        image_width=args.image_size,
        pretrained_path=args.pretrained
    )
    model.to(device)
    print(f"   ✅ Модель создана")
    
    # Подсчитываем параметры
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   📊 Параметров: {total_params:,} (обучаемых: {trainable_params:,})")
    print()
    
    # Оптимизатор и планировщик
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Тренировщик
    trainer = KeypointModelTrainer(
        model=model,
        device=device,
        classification_weight=args.classification_weight,
        keypoint_weight=args.keypoint_weight,
        order_penalty_weight=args.order_penalty_weight,
        geometry_penalty_weight=args.geometry_penalty_weight,
        tolerance_normalized=args.tolerance_normalized
    )
    
    # Обучение
    print("🚀 Начало обучения...")
    print()
    
    try:
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            optimizer=optimizer,
            scheduler=scheduler,
            save_dir=str(save_dir),
            save_prefix='keypoint_model'
        )
        
        print("=" * 60)
        print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
        print("=" * 60)
        print()
        print(f"📁 Модели сохранены в: {save_dir}")
        print(f"   • Лучшая модель: keypoint_model_best.pth")
        print(f"   • Последняя модель: keypoint_model_last.pth")
        print()
        
    except KeyboardInterrupt:
        print("\n⚠️  Обучение прервано пользователем")
        print(f"💾 Последняя модель сохранена: {save_dir}/keypoint_model_last.pth")
    except Exception as e:
        print(f"\n❌ Ошибка во время обучения: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


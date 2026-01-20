"""
Архитектура нейронной сети для распознавания паттернов "Флаг" с детекцией ключевых точек
Мультизадачная модель: классификация + регрессия координат точек T0-T4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlagPatternKeypointCNN(nn.Module):
    """
    Сверточная нейронная сеть для классификации паттернов "Флаг" и детекции ключевых точек
    
    Вход: изображение графика свечей (3 канала: OHLC нормализованный)
    Выход 1: вероятность наличия паттерна (бычий/медвежий/нет) - классификация
    Выход 2: координаты ключевых точек T0-T4 - регрессия (10 значений: x,y для каждой точки)
    """
    
    def __init__(self, num_classes=3, num_keypoints=5, image_height=224, image_width=224):
        """
        Args:
            num_classes: Количество классов (0=нет паттерна, 1=бычий, 2=медвежий)
            num_keypoints: Количество ключевых точек (5: T0, T1, T2, T3, T4)
            image_height: Высота входного изображения
            image_width: Ширина входного изображения
        """
        super(FlagPatternKeypointCNN, self).__init__()
        
        self.num_classes = num_classes
        self.num_keypoints = num_keypoints
        self.image_height = image_height
        self.image_width = image_width
        
        # Первый блок сверток
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Второй блок сверток
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Третий блок сверток
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Четвертый блок сверток
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Вычисляем размер после сверток
        # После 4 пулов: 224 -> 112 -> 56 -> 28 -> 14
        self.fc_input_size = 256 * (image_height // 16) * (image_width // 16)
        
        # Общие полносвязные слои (shared features)
        self.fc_shared = nn.Linear(self.fc_input_size, 512)
        self.dropout_shared = nn.Dropout(0.5)
        
        # Ветка классификации
        self.fc_class1 = nn.Linear(512, 256)
        self.dropout_class = nn.Dropout(0.5)
        self.fc_class2 = nn.Linear(256, num_classes)
        
        # Ветка регрессии ключевых точек
        self.fc_keypoint1 = nn.Linear(512, 256)
        self.dropout_keypoint = nn.Dropout(0.5)
        self.fc_keypoint2 = nn.Linear(256, num_keypoints * 2)  # x, y для каждой точки
    
    def forward(self, x):
        """Прямой проход"""
        # CNN backbone (общие признаки)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool4(x)
        
        # Выпрямление
        x = x.view(x.size(0), -1)
        
        # Общие признаки
        shared_features = F.relu(self.fc_shared(x))
        shared_features = self.dropout_shared(shared_features)
        
        # Ветка классификации
        class_features = F.relu(self.fc_class1(shared_features))
        class_features = self.dropout_class(class_features)
        class_logits = self.fc_class2(class_features)
        
        # Ветка регрессии ключевых точек
        keypoint_features = F.relu(self.fc_keypoint1(shared_features))
        keypoint_features = self.dropout_keypoint(keypoint_features)
        keypoints = self.fc_keypoint2(keypoint_features)  # [batch, num_keypoints * 2]
        keypoints = keypoints.view(-1, self.num_keypoints, 2)  # [batch, num_keypoints, 2]
        
        return class_logits, keypoints
    
    def predict(self, x):
        """Предсказание с применением softmax для классификации"""
        with torch.no_grad():
            class_logits, keypoints = self.forward(x)
            probabilities = F.softmax(class_logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        return predicted_class, probabilities, keypoints


def create_keypoint_model(num_classes=3, num_keypoints=5, image_height=224, image_width=224, pretrained_path=None):
    """
    Создает модель с keypoint detection, опционально загружая предобученные веса
    
    Args:
        num_classes: Количество классов
        num_keypoints: Количество ключевых точек (5: T0-T4)
        image_height: Высота изображения
        image_width: Ширина изображения
        pretrained_path: Путь к файлу с весами для загрузки
    
    Returns:
        Объект модели
    """
    model = FlagPatternKeypointCNN(num_classes, num_keypoints, image_height, image_width)
    
    if pretrained_path:
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print(f"✅ Загружены веса из {pretrained_path}")
        except Exception as e:
            print(f"⚠️ Не удалось загрузить веса: {e}")
            print("   Создается новая модель")
    
    return model


if __name__ == "__main__":
    # Тестирование архитектуры
    model = FlagPatternKeypointCNN(num_classes=3, num_keypoints=5)
    
    # Тестовый вход (batch_size=2, channels=3, height=224, width=224)
    test_input = torch.randn(2, 3, 224, 224)
    
    # Прямой проход
    class_logits, keypoints = model(test_input)
    print(f"Размер выхода классификации: {class_logits.shape}")  # [2, 3]
    print(f"Размер выхода ключевых точек: {keypoints.shape}")  # [2, 5, 2]
    
    # Предсказание
    pred_class, probabilities, keypoints_pred = model.predict(test_input)
    print(f"Предсказанные классы: {pred_class}")
    print(f"Вероятности: {probabilities}")
    print(f"Предсказанные ключевые точки: {keypoints_pred}")


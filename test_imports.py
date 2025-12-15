#!/usr/bin/env python3
"""
Скрипт для проверки корректности импортов
"""
import sys
import os

# Добавляем корень проекта в путь
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Корень проекта: {project_root}")
print(f"PYTHONPATH: {sys.path[:3]}...")
print(f"Текущая директория: {os.getcwd()}")
print()

try:
    print("Попытка импорта cult_test.cult_main...")
    from cult_test.cult_main import TradingBot, Config
    print("✅ Успешно импортирован cult_test.cult_main")
except ImportError as e:
    print(f"❌ Ошибка импорта cult_test.cult_main: {e}")
    sys.exit(1)

try:
    print("Попытка импорта cult_test.strategy_optimizer...")
    from cult_test.strategy_optimizer import StrategyOptimizer
    print("✅ Успешно импортирован cult_test.strategy_optimizer")
except ImportError as e:
    print(f"❌ Ошибка импорта cult_test.strategy_optimizer: {e}")
    sys.exit(1)

try:
    print("Попытка импорта t_tech.invest...")
    from t_tech.invest import Client
    print("✅ Успешно импортирован t_tech.invest")
except ImportError as e:
    print(f"❌ Ошибка импорта t_tech.invest: {e}")
    print("Убедитесь, что установлен пакет t-tech-investments: pip install t-tech-investments")
    sys.exit(1)

print()
print("✅ Все импорты успешны!")

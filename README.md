# Cult Trade Bot

Торговый бот для работы с T-Invest API (T-Банк) с использованием стратегии EMA (Exponential Moving Average).

## Структура проекта

- `cult_test/cult_main.py` - основной модуль торгового бота
- `cult_test/dashboard.py` - Streamlit dashboard для визуализации и управления
- `requirements.txt` - зависимости Python

## Установка зависимостей

Все зависимости устанавливаются через pip из `requirements.txt`:

```bash
pip install -r requirements.txt
```

Основные зависимости:
- `t-tech-investments` - официальная библиотека T-Invest API (новый официальный SDK от T-Банка)
- `pandas>=2.0.0` - работа с данными
- `plotly>=5.0.0` - визуализация графиков
- `streamlit>=1.28.0` - веб-интерфейс
- `python-dotenv>=1.0.0` - загрузка переменных окружения

> **Важно:** Пакет `tinkoff-investments` больше не поддерживается. Используется новый официальный пакет `t-tech-investments`. Все импорты обновлены с `tinkoff` на `t_tech`.

## Настройка

1. Создайте файл `.env` в корне проекта:
```bash
TINKOFF_INVEST_TOKEN=your_token_here
```

2. Настройте параметры в `cult_test/cult_main.py` (класс `Config`):
   - `TICKER` - тикер инструмента (по умолчанию "SBER")
   - `CLASS_CODE` - код класса инструмента (по умолчанию "TQBR")
   - `EMA_SHORT` - период короткой EMA (по умолчанию 30)
   - `EMA_LONG` - период длинной EMA (по умолчанию 195)
   - `TRAILING_INDENT` - отступ для trailing stop (по умолчанию 0.3%)
   - `TAKE_PROFIT` - уровень take profit (по умолчанию 1.0%)

## Запуск локально

### Запуск бота:
```bash
python cult_test/cult_main.py
```

### Запуск dashboard:
```bash
streamlit run cult_test/dashboard.py
```

### Проверка импортов:
Если возникают проблемы с импортами, используйте тестовый скрипт:
```bash
python test_imports.py
```
Этот скрипт проверит корректность всех импортов и покажет детальную информацию о путях.

## Запуск в Docker

### Использование Docker Compose (рекомендуется):

#### Быстрый старт

```bash
# 1. Создайте файл .env с токеном
echo "TINKOFF_INVEST_TOKEN=your_token_here" > .env

# 2. Запустите dashboard
docker compose up dashboard

# 3. Откройте http://localhost:8501 в браузере
```

#### Подготовка

1. Убедитесь, что файл `.env` создан в корне проекта и содержит токен:
```bash
TINKOFF_INVEST_TOKEN=your_token_here
```

2. Убедитесь, что Docker и Docker Compose установлены:
```bash
docker --version
docker-compose --version
```

#### Первый запуск (сборка образов)

```bash
# Собрать образы и запустить все сервисы
docker-compose up --build

# Или собрать образы без запуска
docker-compose build
```

#### Запуск сервисов

**Запуск только dashboard (веб-интерфейс):**
```bash
docker-compose up dashboard
```
После запуска откройте браузер и перейдите на `http://localhost:8501`

**Запуск только бота (торговый бот):**
```bash
docker-compose up bot
```

**Запуск обоих сервисов одновременно:**
```bash
docker-compose up
```

**Запуск в фоновом режиме (detached mode):**
```bash
# Все сервисы
docker-compose up -d

# Только dashboard
docker-compose up -d dashboard

# Только бот
docker-compose up -d bot
```

#### Управление сервисами

**Остановка сервисов:**
```bash
# Остановить все сервисы
docker-compose stop

# Остановить конкретный сервис
docker-compose stop dashboard
docker-compose stop bot
```

**Остановка и удаление контейнеров:**
```bash
docker-compose down
```

**Перезапуск сервисов:**
```bash
# Перезапустить все сервисы
docker-compose restart

# Перезапустить конкретный сервис
docker-compose restart dashboard
```

**Просмотр логов:**
```bash
# Логи всех сервисов
docker-compose logs

# Логи конкретного сервиса
docker-compose logs dashboard
docker-compose logs bot

# Логи в реальном времени (follow)
docker-compose logs -f dashboard
docker-compose logs -f bot
```

**Просмотр статуса сервисов:**
```bash
docker-compose ps
```

**Пересборка образов:**
```bash
# Пересобрать образы без кеша
docker-compose build --no-cache

# Пересобрать и перезапустить
docker-compose up --build --force-recreate
```

#### Полезные команды

**Выполнить команду внутри контейнера:**
```bash
# Войти в контейнер dashboard
docker-compose exec dashboard bash

# Выполнить команду в контейнере бота
docker-compose exec bot python cult_test/cult_main.py
```

**Просмотр использования ресурсов:**
```bash
docker-compose top
```

**Очистка:**
```bash
# Остановить и удалить контейнеры, сети
docker-compose down

# Удалить также volumes
docker-compose down -v

# Удалить образы
docker-compose down --rmi all
```

### Использование Docker напрямую:

1. Соберите образ:
```bash
docker build -t cult_trade_bot .
```

2. Запуск dashboard:
```bash
docker run -p 8501:8501 --env-file .env -e RUN_MODE=dashboard cult_trade_bot
```

3. Запуск бота:
```bash
docker run --env-file .env -e RUN_MODE=bot cult_trade_bot
```

## Структура Docker

- `Dockerfile` - конфигурация Docker образа
- `docker-compose.yml` - конфигурация для Docker Compose
- `.dockerignore` - файлы, исключаемые из Docker образа

## Примечания

- **Новый официальный SDK:** Библиотека `t-tech-investments` - это новый официальный SDK Python для T-Инвестиций от T-Банка
- **Миграция с tinkoff:** Старый пакет `tinkoff-investments` больше не поддерживается. Все импорты обновлены:
  - `from tinkoff.invest import` → `from t_tech.invest import`
  - `import tinkoff` → `import t_tech`
- Локальная директория `tinkoff/` была удалена, так как библиотека доступна через pip
- Все зависимости указаны в `requirements.txt`
- Документация: https://developer.tbank.ru/invest/sdk/python_sdk/

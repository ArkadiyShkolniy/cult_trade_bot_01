# Используем официальный Python образ
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Добавляем корень проекта в PYTHONPATH для корректных импортов
ENV PYTHONPATH=/app

# Устанавливаем системные зависимости для streamlit и других библиотек
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Открываем порт для Streamlit
EXPOSE 8501

# Переменная окружения для выбора режима запуска (dashboard или bot)
# По умолчанию запускаем dashboard
ENV RUN_MODE=dashboard

# Команда по умолчанию - запуск Streamlit dashboard
# Можно переопределить через docker-compose или docker run:
# - Для dashboard: RUN_MODE=dashboard (по умолчанию)
# - Для бота: RUN_MODE=bot
# Запускаем streamlit из корня проекта для корректных импортов
CMD ["sh", "-c", "if [ \"$RUN_MODE\" = \"bot\" ]; then python cult_test/cult_main.py; else cd /app && streamlit run cult_test/dashboard.py --server.port=8501 --server.address=0.0.0.0; fi"]

import streamlit as st
import pandas as pd
import time
import sys
import io
import logging
from contextlib import redirect_stdout

# Импортируем вашего бота и конфиг
# Обратите внимание на путь импорта, он зависит от того, откуда запускаете streamlit
try:
    from cult_test.cult_main import TradingBot, Config
except ImportError:
    st.error("Не удалось найти файл cult_main.py. Запускайте streamlit из корня проекта.")
    st.stop()

# Настройка страницы
st.set_page_config(page_title="Tinkoff Trading Bot", layout="wide")
st.title("🤖 Панель управления торговым роботом")

# --- Боковая панель с настройками ---
st.sidebar.header("Настройки стратегии")

# Параметры инструмента
ticker = st.sidebar.text_input("Тикер инструмента", value=Config.TICKER)
class_code = st.sidebar.text_input("Class Code", value=Config.CLASS_CODE)

# Параметры EMA
st.sidebar.subheader("Индикаторы EMA")
ema_short = st.sidebar.number_input("EMA Короткая", min_value=1, value=Config.EMA_SHORT)
ema_long = st.sidebar.number_input("EMA Длинная", min_value=1, value=Config.EMA_LONG)

# Параметры Риск-менеджмента
st.sidebar.subheader("Риск-менеджмент (%)")
take_profit = st.sidebar.number_input("Take Profit", min_value=0.1, value=Config.TAKE_PROFIT * 100, step=0.1)
trailing_indent = st.sidebar.number_input("Trailing Indent (отступ)", min_value=0.01, value=Config.TRAILING_INDENT * 100, step=0.01)
trailing_spread = st.sidebar.number_input("Trailing Spread (защита)", min_value=0.0, value=Config.TRAILING_SPREAD, step=0.1)

# Обновление конфигурации
# Мы меняем атрибуты класса Config напрямую перед запуском бота
Config.TICKER = ticker
Config.CLASS_CODE = class_code
Config.EMA_SHORT = int(ema_short)
Config.EMA_LONG = int(ema_long)
Config.TAKE_PROFIT = take_profit / 100
Config.TRAILING_INDENT = trailing_indent / 100
Config.TRAILING_SPREAD = trailing_spread

# --- Основная область ---

col1, col2 = st.columns(2)

with col1:
    st.subheader("Управление")
    # Состояние запуска
    if 'running' not in st.session_state:
        st.session_state.running = False

    start_btn = st.button("🚀 Запустить одну итерацию", type="primary")
    loop_check = st.checkbox("Циклический запуск (каждые 60 сек)")

    if start_btn:
        st.session_state.running = True

with col2:
    st.subheader("Статус")
    status_placeholder = st.empty()

# Область логов
st.subheader("Лог операций")
log_container = st.container()
log_text = st.empty()

# --- Логика запуска ---

class StreamlitLogger(logging.Handler):
    """Кастомный обработчик логов для вывода в Streamlit"""
    def __init__(self, widget):
        super().__init__()
        self.widget = widget
        self.log_buffer = []

    def emit(self, record):
        msg = self.format(record)
        self.log_buffer.append(msg)
        # Оставляем последние 50 строк
        if len(self.log_buffer) > 50:
            self.log_buffer.pop(0)
        self.widget.code("\n".join(self.log_buffer))

# Настраиваем перехват логов
if 'logger_setup' not in st.session_state:
    root_logger = logging.getLogger()
    # Удаляем старые хендлеры, чтобы не дублировать
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    
    # Добавляем наш хендлер
    st_handler = StreamlitLogger(log_text)
    st_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(st_handler)
    root_logger.setLevel(logging.INFO)
    st.session_state.logger_setup = True


if st.session_state.running:
    try:
        status_placeholder.info("⏳ Бот работает...")
        
        # Инициализация и запуск бота
        bot = TradingBot()
        bot.run()
        
        status_placeholder.success("✅ Итерация завершена")
        
        if not loop_check:
            st.session_state.running = False
        else:
            time.sleep(60) # Пауза перед следующим циклом
            st.rerun() # Перезапуск скрипта для следующей итерации
            
    except Exception as e:
        status_placeholder.error(f"❌ Ошибка: {e}")
        st.session_state.running = False

import streamlit as st
import pandas as pd
import time
import logging
import sys
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Добавляем корень проекта в путь для импортов
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Импортируем вашего бота и конфиг
try:
    from cult_test.cult_main import TradingBot, Config
    from t_tech.invest import Client
except ImportError as e:
    st.error(f"Ошибка импорта: {e}")
    st.error(f"PYTHONPATH: {sys.path}")
    st.error(f"Текущая директория: {os.getcwd()}")
    st.error(f"Файл dashboard.py: {__file__}")
    st.error("Убедитесь, что запускаете streamlit из корня проекта.")
    st.stop()

# Настройка страницы
st.set_page_config(page_title="Trading Bot Dashboard", layout="wide", page_icon="📈")
st.title("📈 Панель управления и Аналитика")

# --- Боковая панель с настройками ---
st.sidebar.header("⚙️ Настройки")

# Параметры инструмента
ticker = st.sidebar.text_input("Тикер", value=Config.TICKER)
class_code = st.sidebar.text_input("Class Code", value=Config.CLASS_CODE)

# Параметры EMA
st.sidebar.subheader("Индикаторы")
ema_short_val = st.sidebar.number_input("EMA Fast", min_value=1, value=Config.EMA_SHORT)
ema_long_val = st.sidebar.number_input("EMA Slow", min_value=1, value=Config.EMA_LONG)

# Параметры Риск-менеджмента
st.sidebar.subheader("Риск-менеджмент")
take_profit = st.sidebar.number_input("Take Profit (%)", min_value=0.1, value=Config.TAKE_PROFIT * 100, step=0.1)
trailing_indent = st.sidebar.number_input("Trailing Indent (%)", min_value=0.01, value=Config.TRAILING_INDENT * 100, step=0.01)

# Обновление глобальной конфигурации
Config.TICKER = ticker
Config.CLASS_CODE = class_code
Config.EMA_SHORT = int(ema_short_val)
Config.EMA_LONG = int(ema_long_val)
Config.TAKE_PROFIT = take_profit / 100
Config.TRAILING_INDENT = trailing_indent / 100

# --- Функция загрузки данных для графика ---
@st.cache_data(ttl=60)  # Кешируем данные на 60 секунд
def load_market_data(ticker, short_period, long_period):
    """Загружает свечи и считает индикаторы для визуализации"""
    bot = TradingBot()
    
    try:
        # Используем клиент только для чтения данных
        with Client(Config.TOKEN) as client:
            bot.client = client
            bot._setup_instrument() # Находим инструмент по тикеру
            
            # Загружаем свечи (берем больше данных для корректного расчета EMA)
            df = bot._get_candles_dataframe(days_back=60)
            
            if df.empty:
                return None, "Нет данных по свечам"
                
            # Расчет индикаторов (дублируем логику бота для визуализации)
            df['ema_short'] = df['close'].ewm(span=short_period, adjust=False).mean()
            df['ema_long'] = df['close'].ewm(span=long_period, adjust=False).mean()
            
            return df, None
    except Exception as e:
        return None, str(e)

# --- Основной экран ---

# 1. Секция Графика
st.subheader(f"График {ticker} (30 min)")

# Кнопка обновления графика
if st.button("🔄 Обновить данные рынка"):
    load_market_data.clear()

with st.spinner("Загрузка рыночных данных..."):
    df, error = load_market_data(ticker, Config.EMA_SHORT, Config.EMA_LONG)

if error:
    st.error(f"Ошибка загрузки данных: {error}")
elif df is not None:
    # Отрисовка с Plotly
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    # Свечи
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Цена'
    ))

    # EMA Short
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['ema_short'],
        line=dict(color='orange', width=1.5),
        name=f'EMA {Config.EMA_SHORT}'
    ))

    # EMA Long
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['ema_long'],
        line=dict(color='blue', width=1.5),
        name=f'EMA {Config.EMA_LONG}'
    ))

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=10, r=10, t=30, b=10),
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Показываем последние значения
    last_candle = df.iloc[-1]
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Цена закрытия", f"{last_candle['close']:.2f}")
    col_m2.metric(f"EMA {Config.EMA_SHORT}", f"{last_candle['ema_short']:.2f}")
    col_m3.metric(f"EMA {Config.EMA_LONG}", f"{last_candle['ema_long']:.2f}", 
                  delta=f"{last_candle['ema_short'] - last_candle['ema_long']:.2f}")

st.divider()

# 2. Секция Управления Ботом
col_ctrl, col_log = st.columns([1, 2])

with col_ctrl:
    st.subheader("🤖 Управление ботом")
    
    if 'running' not in st.session_state:
        st.session_state.running = False

    start_btn = st.button("🚀 Запустить проверку", type="primary", use_container_width=True)
    loop_check = st.checkbox("Авто-режим (каждые 60 сек)")
    
    if start_btn:
        st.session_state.running = True

with col_log:
    st.subheader("Лог операций")
    log_text = st.empty()

# --- Логика запуска и логирования ---

class StreamlitLogger(logging.Handler):
    def __init__(self, widget):
        super().__init__()
        self.widget = widget
        self.log_buffer = []

    def emit(self, record):
        msg = self.format(record)
        # Добавляем эмодзи для красоты логов
        if "BUY" in msg: msg = "🟢 " + msg
        elif "SELL" in msg: msg = "🔴 " + msg
        elif "HOLD" in msg: msg = "⚪ " + msg
        
        self.log_buffer.append(msg)
        if len(self.log_buffer) > 20: # Держим последние 20 строк
            self.log_buffer.pop(0)
        self.widget.code("\n".join(self.log_buffer), language="text")

# Настройка логгера
if 'logger_setup' not in st.session_state:
    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    
    st_handler = StreamlitLogger(log_text)
    st_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
    root_logger.addHandler(st_handler)
    root_logger.setLevel(logging.INFO)
    st.session_state.logger_setup = True

# Запуск логики бота
if st.session_state.running:
    try:
        with st.spinner("Анализ рынка и проверка условий..."):
            bot = TradingBot()
            bot.run()
        
        if not loop_check:
            st.session_state.running = False
            st.success("Проверка завершена")
        else:
            time.sleep(60)
            st.rerun()
            
    except Exception as e:
        st.error(f"Ошибка исполнения: {e}")
        st.session_state.running = False

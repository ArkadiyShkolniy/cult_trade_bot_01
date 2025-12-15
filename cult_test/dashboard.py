import streamlit as st
import pandas as pd
import time
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Импортируем бота и конфиг
try:
    from cult_test.cult_main import TradingBot, Config
    from t_tech.invest import Client
    from t_tech.invest.utils import quotation_to_decimal
except ImportError as e:
    st.error(f"Ошибка импорта: {e}. Убедитесь, что запускаете из корня проекта.")
    st.stop()

# Настройка страницы
st.set_page_config(
    page_title="Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
)

st.title(f"🤖 Trading Bot Dashboard: {Config.TICKER}")

# --- Боковая панель: Управление ---
st.sidebar.header("⚙️ Управление")

# Статус бота (эмуляция)
if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False

def toggle_bot():
    st.session_state.bot_running = not st.session_state.bot_running

btn_label = "⛔ Остановить бота" if st.session_state.bot_running else "▶️ Запустить бота"
st.sidebar.button(btn_label, on_click=toggle_bot)

if st.session_state.bot_running:
    st.sidebar.success("Статус: РАБОТАЕТ")
else:
    st.sidebar.warning("Статус: ОСТАНОВЛЕН")

st.sidebar.markdown("---")
st.sidebar.header("🛠 Параметры стратегии")

# Чтение и изменение параметров Config (влияет на текущую сессию)
ema_short = st.sidebar.number_input("EMA Short", min_value=5, value=Config.EMA_SHORT)
ema_long = st.sidebar.number_input("EMA Long", min_value=20, value=Config.EMA_LONG)

# Обновляем конфиг при изменении
if ema_short != Config.EMA_SHORT:
    Config.EMA_SHORT = ema_short
if ema_long != Config.EMA_LONG:
    Config.EMA_LONG = ema_long

# --- Основная часть: График ---
st.subheader("График и Индикаторы")

@st.cache_data(ttl=60)  # Кэшируем данные на 60 секунд
def load_market_data():
    """Загрузка данных через методы бота"""
    try:
        bot = TradingBot()
        # Инициализируем клиента для загрузки данных (без полной настройки аккаунта)
        with Client(Config.TOKEN) as client:
            bot.client = client
            bot._setup_instrument()
            df = bot._get_candles_dataframe(days_back=10) # 10 дней для графика
            return df, None
    except Exception as e:
        return None, str(e)

with st.spinner('Загрузка данных рынка...'):
    df, error = load_market_data()

if error:
    st.error(f"Ошибка загрузки данных: {error}")
elif df is not None and not df.empty:
    # Расчет индикаторов для отображения
    df['ema_short'] = df['close'].ewm(span=Config.EMA_SHORT, adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=Config.EMA_LONG, adjust=False).mean()

    # Последняя свеча
    last_close = df.iloc[-1]['close']
    last_ema_s = df.iloc[-1]['ema_short']
    last_ema_l = df.iloc[-1]['ema_long']

    # Метрики
    col1, col2, col3 = st.columns(3)
    col1.metric("Цена Close", f"{last_close:.2f}")
    col2.metric(f"EMA {Config.EMA_SHORT}", f"{last_ema_s:.2f}", delta=f"{last_ema_s - last_ema_l:.2f}")
    col3.metric(f"EMA {Config.EMA_LONG}", f"{last_ema_l:.2f}")

    # График Plotly
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.05)

    # Свечи
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'] if 'open' in df.columns else df['close'], # Fallback если только close
        high=df['high'] if 'high' in df.columns else df['close'],
        low=df['low'] if 'low' in df.columns else df['close'],
        close=df['close'],
        name='Цена'
    ))

    # EMA Линии
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_short'], line=dict(color='orange', width=1), name=f'EMA {Config.EMA_SHORT}'))
    fig.add_trace(go.Scatter(x=df.index, y=df['ema_long'], line=dict(color='blue', width=1), name=f'EMA {Config.EMA_LONG}'))

    # Сигналы (точки пересечения)
    # Находим точки пересечения
    cross_buy = df[(df['ema_short'] > df['ema_long']) & (df['ema_short'].shift(1) <= df['ema_long'].shift(1))]
    cross_sell = df[(df['ema_short'] < df['ema_long']) & (df['ema_short'].shift(1) >= df['ema_long'].shift(1))]

    fig.add_trace(go.Scatter(
        x=cross_buy.index, y=cross_buy['ema_short'],
        mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'),
        name='Signal BUY'
    ))
    
    fig.add_trace(go.Scatter(
        x=cross_sell.index, y=cross_sell['ema_short'],
        mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'),
        name='Signal SELL'
    ))

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=10, r=10, t=30, b=10),
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("Нет данных для отображения.")

# Автообновление
if st.session_state.bot_running:
    time.sleep(60)
    st.rerun()

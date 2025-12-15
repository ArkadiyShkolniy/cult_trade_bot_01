import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Импортируем бота и конфиг
try:
    from cult_test.cult_main import TradingBot, Config
    from t_tech.invest import Client
except ImportError as e:
    st.error(f"Ошибка импорта: {e}")
    st.stop()

st.set_page_config(page_title="Trading Bot Dashboard", layout="wide")
st.title(f"🤖 Trading Bot: {Config.TICKER}")

# --- Боковая панель ---
st.sidebar.header("⚙️ Управление")

if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False

def toggle_bot():
    st.session_state.bot_running = not st.session_state.bot_running

btn_label = "⛔ Остановить" if st.session_state.bot_running else "▶️ Запустить"
st.sidebar.button(btn_label, on_click=toggle_bot)

# Кнопка сброса кэша и обновления
if st.sidebar.button("🔄 Обновить данные"):
    st.cache_data.clear()
    st.rerun()

if st.session_state.bot_running:
    st.sidebar.success("РАБОТАЕТ")
else:
    st.sidebar.warning("ОСТАНОВЛЕН")

# Параметры
st.sidebar.markdown("---")
Config.EMA_SHORT = st.sidebar.number_input("EMA Short", min_value=5, value=Config.EMA_SHORT)
Config.EMA_LONG = st.sidebar.number_input("EMA Long", min_value=20, value=Config.EMA_LONG)

# --- График ---
st.subheader("График (M15)")

@st.cache_data(ttl=60)
def load_data():
    try:
        bot = TradingBot()
        with Client(Config.TOKEN) as client:
            bot.client = client
            bot._setup_instrument()
            # ВАЖНО: Загружаем данные за 40 дней для корректного расчета EMA 260
            return bot._get_candles_dataframe(days_back=40), None
    except Exception as e:
        return None, str(e)

with st.spinner('Загрузка...'):
    df_full, error = load_data()

if df_full is not None and not df_full.empty:
    # 1. Сначала считаем индикаторы на ПОЛНОЙ истории
    df_full['ema_short'] = df_full['close'].ewm(span=Config.EMA_SHORT, adjust=False).mean()
    df_full['ema_long'] = df_full['close'].ewm(span=Config.EMA_LONG, adjust=False).mean()

    # 2. Обрезаем для отображения (последние ~500 свечей)
    df = df_full.tail(500).copy()

    # Форматируем дату
    df['date_str'] = df.index.strftime('%d.%m %H:%M')

    # Метрики
    last = df.iloc[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("Цена", f"{last['close']:.2f}")
    c2.metric(f"EMA {Config.EMA_SHORT}", f"{last['ema_short']:.2f}")
    c3.metric(f"EMA {Config.EMA_LONG}", f"{last['ema_long']:.2f}")

    # График
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    # Свечи
    fig.add_trace(go.Candlestick(
        x=df['date_str'],
        open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='Цена',
        increasing_line_color='#26A69A', decreasing_line_color='#EF5350'
    ))

    # EMA
    fig.add_trace(go.Scatter(x=df['date_str'], y=df['ema_short'], line=dict(color='#FFA726', width=1.5), name=f'EMA {Config.EMA_SHORT}'))
    fig.add_trace(go.Scatter(x=df['date_str'], y=df['ema_long'], line=dict(color='#42A5F5', width=1.5), name=f'EMA {Config.EMA_LONG}'))

    # Сигналы
    cross_buy = df[(df['ema_short'] > df['ema_long']) & (df['ema_short'].shift(1) <= df['ema_long'].shift(1))]
    cross_sell = df[(df['ema_short'] < df['ema_long']) & (df['ema_short'].shift(1) >= df['ema_long'].shift(1))]

    if not cross_buy.empty:
        fig.add_trace(go.Scatter(
            x=cross_buy['date_str'], y=cross_buy['ema_short'],
            mode='markers', marker=dict(color='#00E676', size=12, symbol='triangle-up'),
            name='BUY'
        ))
    if not cross_sell.empty:
        fig.add_trace(go.Scatter(
            x=cross_sell['date_str'], y=cross_sell['ema_short'],
            mode='markers', marker=dict(color='#FF1744', size=12, symbol='triangle-down'),
            name='SELL'
        ))

    # Настройки
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_dark",
        xaxis=dict(
            type='category', 
            nticks=10, 
            tickangle=-45
        ),
        legend=dict(orientation="h", y=1, x=0)
    )

    st.plotly_chart(fig, use_container_width=True)

elif error:
    st.error(f"Ошибка: {error}")
else:
    st.info("Нет данных")

if st.session_state.bot_running:
    time.sleep(60)
    st.rerun()
